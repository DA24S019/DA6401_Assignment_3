import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CharTokenizer:
    def __init__(self):
        self.special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        self.char2idx = self.special_tokens.copy()
        chars = set()
        for text in texts:
            chars.update(text)
        for char in sorted(chars):
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

class TransliterationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src = src_sequences
        self.tgt = tgt_sequences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def load_data(batch_size):
    base_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
    train_df = pd.read_csv(os.path.join(base_dir, 'hi.translit.sampled.train.tsv'), sep='\t', names=['devanagari', 'latin', 'freq'])
    val_df = pd.read_csv(os.path.join(base_dir, 'hi.translit.sampled.dev.tsv'), sep='\t', names=['devanagari', 'latin', 'freq'])

    for col in ['latin', 'devanagari']:
        train_df[col] = train_df[col].fillna('').astype(str).str.strip()
        val_df[col] = val_df[col].fillna('').astype(str).str.strip()

    input_tokenizer = CharTokenizer()
    input_tokenizer.build_vocab(train_df['latin'])
    output_tokenizer = CharTokenizer()
    output_tokenizer.build_vocab(train_df['devanagari'])

    def process_sequences(words, tokenizer):
        sequences = []
        for word in words:
            seq = [tokenizer.char2idx['<sos>']]
            seq += [tokenizer.char2idx.get(c, 0) for c in word]
            seq += [tokenizer.char2idx['<eos>']]
            sequences.append(torch.tensor(seq, dtype=torch.long))
        return sequences

    train_src = pad_sequence(process_sequences(train_df['latin'], input_tokenizer), batch_first=True)
    train_tgt = pad_sequence(process_sequences(train_df['devanagari'], output_tokenizer), batch_first=True)
    val_src = pad_sequence(process_sequences(val_df['latin'], input_tokenizer), batch_first=True)
    val_tgt = pad_sequence(process_sequences(val_df['devanagari'], output_tokenizer), batch_first=True)

    train_loader = DataLoader(TransliterationDataset(train_src, train_tgt), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TransliterationDataset(val_src, val_tgt), batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, input_tokenizer, output_tokenizer
