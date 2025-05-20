import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_vocab_size: int
    output_vocab_size: int
    embedding_size: int
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    cell_type: str
    dropout: float

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch, seq_len, hidden_size]
        # decoder_hidden: [num_layers, batch, hidden_size] or [batch, hidden_size]
        if decoder_hidden.dim() == 3:
            # Use last layer's hidden state
            decoder_hidden = decoder_hidden[-1]
        # decoder_hidden: [batch, hidden_size]
        decoder_hidden = decoder_hidden.unsqueeze(1)  # [batch, 1, hidden_size]
        # Compute scores
        scores = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden)))  # [batch, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # [batch, 1, hidden_size]
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.input_vocab_size, config.embedding_size)
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size,
            config.hidden_size,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.output_vocab_size, config.embedding_size)
        self.attention = BahdanauAttention(config.hidden_size)
        
        rnn_map = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.rnn = rnn_map[config.cell_type](
            config.embedding_size + config.hidden_size,  # Input size changed
            config.hidden_size,
            num_layers=config.decoder_layers,
            dropout=config.dropout if config.decoder_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, config.output_vocab_size)

    def forward(self, x, hidden, encoder_outputs, return_attention=False):
        embedded = self.embedding(x)
        if isinstance(hidden, tuple):  # LSTM
            attn_hidden = hidden[0]
        else:
            attn_hidden = hidden
        context, attn_weights = self.attention(encoder_outputs, attn_hidden)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output)
        if return_attention:
            return prediction, hidden, attn_weights
        else:
            return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def _adjust_hidden(self, hidden, target_layers):
        current_layers = hidden.size(0)
        if current_layers == target_layers:
            return hidden
        elif current_layers < target_layers:
            # Repeat last layer to match decoder layers
            diff = target_layers - current_layers
            return torch.cat([hidden, hidden[-1:].repeat(diff, 1, 1)], dim=0)
        else:
            # Trim extra layers
            return hidden[:target_layers]
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Handle LSTM case explicitly
        if isinstance(encoder_hidden, tuple):
            h, c = encoder_hidden
            h = self._adjust_hidden(h, self.decoder.rnn.num_layers)
            c = self._adjust_hidden(c, self.decoder.rnn.num_layers)
            decoder_hidden = (h, c)
        else:
            decoder_hidden = self._adjust_hidden(encoder_hidden, self.decoder.rnn.num_layers)
        
        # Rest of forward pass remains unchanged
        outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input, decoder_hidden, encoder_outputs)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t].unsqueeze(1) if teacher_force else output.argmax(-1)
            
        return outputs



    def _adjust_hidden(self, hidden, target_layers):
        current_layers = hidden.size(0)
        # print(f"Adjusting hidden from {current_layers} to {target_layers} layers")  # Debug
        if current_layers == target_layers:
            return hidden
        elif current_layers < target_layers:
            diff = target_layers - current_layers
            return torch.cat([hidden, hidden[-1:].repeat(diff, 1, 1)], dim=0)
        else:
            return hidden[:target_layers]



# ======================
# 3. Training Components
# ======================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            self.counter = 0
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: EarlyStopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping triggered')
                self.early_stop = True


class Translator:
    def __init__(self, model_path='best_model_attention.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'  # force CPU if you want, else remove this line
        
        checkpoint = torch.load(model_path, map_location=device)

        class TempConfig:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)
        
        config = TempConfig(checkpoint['config'])

        # Assume Encoder, Decoder, Seq2Seq are defined/imported elsewhere
        encoder = Encoder(config)
        decoder = Decoder(config)
        self.model = Seq2Seq(encoder, decoder, device).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.device = device
        self.input_tokenizer = checkpoint['input_tokenizer']
        self.output_tokenizer = {v: k for k, v in checkpoint['output_tokenizer'].items()}

    def translate_with_attention(self, word, max_length=20):
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
            
            input_tensor = torch.tensor([[self.input_tokenizer['<sos>']]], device=self.device)
            output_chars = []
            attn_matrix = []

            for _ in range(max_length):
                # Make sure decoder returns attention weights when return_attention=True
                output, hidden, attn_weights = self.model.decoder(input_tensor, hidden, encoder_outputs, return_attention=True)
                attn_matrix.append(attn_weights.squeeze(0).cpu().numpy())  # shape: [src_seq_len]
                pred_token = output.argmax(-1).item()
                char = self.output_tokenizer.get(pred_token, '<unk>')
                if char == '<eos>':
                    break
                output_chars.append(char)
                input_tensor = torch.tensor([[pred_token]], device=self.device)
        
        attn_matrix = np.stack(attn_matrix, axis=0)  # shape: [tgt_seq_len, src_seq_len]
        return ''.join(output_chars), attn_matrix


import pandas as pd
import streamlit as st
st.set_page_config(page_title="Hindi Transliteration", page_icon="📝", layout="centered")

# Load test set
@st.cache_data
def load_test_set():
    df = pd.read_csv(
        'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',
        sep='\t', names=['devanagari', 'latin', 'freq'])
    df['latin'] = df['latin'].fillna('').astype(str).str.strip()
    df['devanagari'] = df['devanagari'].fillna('').astype(str).str.strip()
    return df

# Load translator once
@st.cache_resource
def load_translator():
    return Translator('best_model_attention.pth')

translator = load_translator()
test_df = load_test_set()

# Page config

# Title & Description
st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>Hindi Transliteration by Prefix</h1>
    <p style='text-align: center; font-size:16px;'>
        Select a Latin word and choose how many starting letters to transliterate into Devanagari script.
        The prediction updates automatically.
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Inputs
with st.sidebar:
    st.header("Input Options")
    selected_word = st.selectbox("Select a Latin word:", test_df['latin'].tolist())
    
    if selected_word:
        prefix_len = st.slider(
            "Select prefix length:", 
            min_value=1, 
            max_value=len(selected_word), 
            value=len(selected_word),
            help="Choose how many starting letters to transliterate"
        )
    else:
        prefix_len = 1  # fallback

# Main output area
if selected_word:
    prefix = selected_word[:prefix_len]
    st.info(f"**Selected prefix:** {prefix}")
    
    output_word, _ = translator.translate_with_attention(prefix)
    st.success(f"**Predicted Devanagari:** {output_word}")

    # Optional: show full word and length for context
    st.markdown(f"<small style='color:gray;'>Full word: {selected_word} | Prefix length: {prefix_len}</small>", unsafe_allow_html=True)
else:
    st.warning("Please select a Latin word from the sidebar.")

# Footer / credits
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>Developed for DA6401 JAN-JULY 2025</p>", unsafe_allow_html=True)