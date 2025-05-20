import torch
import pandas as pd
import os
import time
from tqdm import tqdm
from model import Seq2Seq, Encoder, Decoder

# ---------- (1) Load Test Data ----------
def load_test_data(tokenizer_path='best_model.pth', limit=None):
    checkpoint = torch.load(tokenizer_path, map_location='cpu')
    
    base_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
    test_path = os.path.join(base_dir, 'hi.translit.sampled.test.tsv')
    if not os.path.exists(test_path):
        test_path += '.gz'

    test_df = pd.read_csv(test_path, sep='\t', 
                          names=['devanagari', 'latin', 'freq'],
                          compression='gzip' if test_path.endswith('.gz') else None)

    test_df['latin'] = test_df['latin'].fillna('').astype(str).str.strip()
    test_df['devanagari'] = test_df['devanagari'].fillna('').astype(str).str.strip()

    if limit is not None:
        test_df = test_df.head(limit)

    return test_df['latin'].tolist(), test_df['devanagari'].tolist()


# ---------- (2) Translator ----------
class Translator:
    def __init__(self, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load config
        class TempConfig:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)

        config = TempConfig(checkpoint['config'])

        self.model = Seq2Seq(Encoder(config), Decoder(config), self.device).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.input_tokenizer = checkpoint['input_tokenizer']
        self.output_tokenizer = checkpoint['output_tokenizer']
        self.output_inv_vocab = {v: k for k, v in self.output_tokenizer.items()}
        self.config = config

    def adjust_hidden(self, hidden, target_layers):
        def pad_or_trim(h, target_layers):
            current_layers = h.size(0)
            if current_layers == target_layers:
                return h
            elif current_layers < target_layers:
                last = h[-1:, :, :]
                return torch.cat([h, last.repeat(target_layers - current_layers, 1, 1)], dim=0)
            else:
                return h[:target_layers]
        
        if isinstance(hidden, tuple):
            h, c = hidden
            return pad_or_trim(h, target_layers), pad_or_trim(c, target_layers)
        return pad_or_trim(hidden, target_layers)

    def translate(self, word, max_length=20):
        seq = [1] + [self.input_tokenizer.get(c, 0) for c in word] + [2]  # SOS + tokens + EOS
        src = torch.tensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hidden = self.model.encoder(src)
            hidden = self.adjust_hidden(hidden, self.model.decoder.rnn.num_layers)
            trg = torch.tensor([[1]], device=self.device)  # SOS

            output_chars = []
            for _ in range(max_length):
                output, hidden = self.model.decoder(trg, hidden)
                pred = output.argmax(-1).item()
                char = self.output_inv_vocab.get(pred, '<unk>')
                if char == '<eos>':
                    break
                output_chars.append(char)
                trg = torch.tensor([[pred]], device=self.device)

        return ''.join(output_chars)


# ---------- (3) Evaluation ----------
def evaluate_test_set(limit=None):
    input_words, output_words = load_test_data(limit=limit)
    translator = Translator()

    correct = 0
    total = 0
    results = []

    print(f"🔤 Translating {len(input_words)} test samples...")
    for latin, devanagari in tqdm(zip(input_words, output_words), total=len(input_words)):
        pred = translator.translate(latin)
        results.append((latin, pred, devanagari))
        correct += int(pred == devanagari)
        total += 1

    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2%}")
    return results, accuracy


# ---------- (4) Save Predictions ----------
def save_predictions(results, folder='predictions_vanilla'):
    os.makedirs(folder, exist_ok=True)

    df = pd.DataFrame(results, columns=['Input', 'Prediction', 'Target'])
    df['Correct'] = df['Prediction'] == df['Target']

    # Save all predictions
    df.to_csv(os.path.join(folder, 'predictions.csv'), index=False)
    df[df['Correct']].to_csv(os.path.join(folder, 'true_predictions.csv'), index=False)
    df[~df['Correct']].to_csv(os.path.join(folder, 'false_predictions.csv'), index=False)

    # Print sample
    sample_df = df.sample(10, random_state=42)
    print("\n📋 Sample Predictions:")
    print(sample_df[['Input', 'Prediction', 'Target', 'Correct']].to_markdown(index=False))

    # Save all as .txt
    with open(os.path.join(folder, 'all_predictions.txt'), 'w') as f:
        for _, row in df.iterrows():
            f.write(f"Input: {row['Input']}\nPrediction: {row['Prediction']}\nTarget: {row['Target']}\nCorrect: {row['Correct']}\n\n")


# ---------- (5) Main ----------
if __name__ == "__main__":
    start_time = time.time()

    test_results, test_acc = evaluate_test_set(limit=None)

    print(f"\n⏱️ Total Time Taken: {time.time() - start_time:.2f} seconds")
    save_predictions(test_results)
