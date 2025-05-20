import torch
import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Import your model classes
from model import Encoder, Decoder, Seq2Seq  # Adjust the import if your file name is different


def load_test_data(tokenizer_path='best_model_attention.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    checkpoint = torch.load(tokenizer_path, map_location=device)

    # Load test data
    base_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
    test_path = os.path.join(base_dir, 'hi.translit.sampled.test.tsv')

    if not os.path.exists(test_path):
        test_path += '.gz'

    test_df = pd.read_csv(
        test_path, sep='\t',
        names=['devanagari', 'latin', 'freq'],
        compression='gzip' if test_path.endswith('.gz') else None
    )

    test_df['latin'] = test_df['latin'].fillna('').astype(str).str.strip()
    test_df['devanagari'] = test_df['devanagari'].fillna('').astype(str).str.strip()

    return test_df['latin'].tolist(), test_df['devanagari'].tolist()


class Translator:
    def __init__(self, model_path='best_model_attention.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        checkpoint = torch.load(model_path, map_location=device)

        class TempConfig:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)

        config = TempConfig(checkpoint['config'])

        encoder = Encoder(config)
        decoder = Decoder(config)
        self.model = Seq2Seq(encoder, decoder, device).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.input_tokenizer = checkpoint['input_tokenizer']
        self.output_tokenizer = {v: k for k, v in checkpoint['output_tokenizer'].items()}
        self.device = device

    def translate(self, word, max_length=20):
        seq = [self.input_tokenizer.get('<sos>', 1)]
        seq += [self.input_tokenizer.get(c, self.input_tokenizer.get('<unk>', 0)) for c in word]
        seq.append(self.input_tokenizer.get('<eos>', 2))

        src = torch.tensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.model.encoder(src)

            if isinstance(encoder_hidden, tuple):  # LSTM
                h, c = encoder_hidden
                h = self.model._adjust_hidden(h, self.model.decoder.rnn.num_layers)
                c = self.model._adjust_hidden(c, self.model.decoder.rnn.num_layers)
                hidden = (h, c)
            else:
                hidden = self.model._adjust_hidden(encoder_hidden, self.model.decoder.rnn.num_layers)

            input = torch.tensor([[self.input_tokenizer['<sos>']]], device=self.device)
            output_chars = []

            for _ in range(max_length):
                output, hidden = self.model.decoder(input, hidden, encoder_outputs)
                pred_token = output.argmax(-1).item()
                char = self.output_tokenizer.get(pred_token, '<unk>')

                if char == '<eos>':
                    break

                output_chars.append(char)
                input = torch.tensor([[pred_token]], device=self.device)

        return ''.join(output_chars)


def evaluate_test_set():
    input_words, output_words = load_test_data('best_model_attention.pth')
    translator = Translator('best_model_attention.pth')

    correct = 0
    total = 0
    results = []

    print("Evaluating test set...")
    start_time = time.time()

    for latin, devanagari in tqdm(zip(input_words, output_words), total=len(input_words), desc="Translating"):
        pred = translator.translate(latin)
        results.append((latin, pred, devanagari))
        if pred == devanagari:
            correct += 1
        total += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print(f"Time taken: {elapsed_time:.2f} seconds for {total} samples ({elapsed_time / total:.2f} sec/sample)")

    return results, accuracy


def save_predictions(results, folder='predictions2att'):
    os.makedirs(folder, exist_ok=True)

    df = pd.DataFrame(results, columns=['Input', 'Prediction', 'Target'])
    df.to_csv(os.path.join(folder, 'predictions.csv'), index=False)

    df['Correct'] = df['Prediction'] == df['Target']
    true_df = df[df['Correct']]
    false_df = df[~df['Correct']]

    true_df.to_csv(os.path.join(folder, 'true_predictions.csv'), index=False)
    false_df.to_csv(os.path.join(folder, 'false_predictions.csv'), index=False)

    sample_df = df.sample(10, random_state=42)
    print("\nSample Predictions:")
    print(sample_df[['Input', 'Prediction', 'Target', 'Correct']].to_markdown(index=False))

    with open(os.path.join(folder, 'all_predictions.txt'), 'w') as f:
        for _, row in df.iterrows():
            f.write(f"Input: {row['Input']}\nPrediction: {row['Prediction']}\nTarget: {row['Target']}\nCorrect: {row['Correct']}\n\n")


if __name__ == '__main__':
    test_results, test_acc = evaluate_test_set()
    save_predictions(test_results, folder='predictions2att')
