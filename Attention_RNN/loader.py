import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset import CharTokenizer, TransliterationDataset

def load_data(batch_size):
    base_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
    train_df = pd.read_csv(os.path.join(base_dir, 'hi.translit.sampled.train.tsv'), sep='\t', names=['devanagari', 'latin', 'freq'])
    val_df = pd.read_csv(os.path.join(base_dir, 'hi.translit.sampled.dev.tsv'), sep='\t', names=['devanagari', 'latin', 'freq'])

    for col in ['latin', 'devanagari']:
        train_df[col] = train_df[col].fillna('').astype(str).str.strip()
        val_df[col] = val_df[col].fillna('').astype(str).str.strip()

    input_tokenizer = CharTokenizer()
    input_tokenizer.build_vocab(train_df['latin'].tolist())

    output_tokenizer = CharTokenizer()
    output_tokenizer.build_vocab(train_df['devanagari'].tolist())

    def process_sequences(words, tokenizer):
        sequences = []
        for word in words:
            seq = [tokenizer.char2idx['<sos>']] + \
                  [tokenizer.char2idx.get(c, 0) for c in word] + \
                  [tokenizer.char2idx['<eos>']]
            sequences.append(torch.tensor(seq, dtype=torch.long))
        return sequences

    train_src = pad_sequence(process_sequences(train_df['latin'], input_tokenizer), batch_first=True)
    train_tgt = pad_sequence(process_sequences(train_df['devanagari'], output_tokenizer), batch_first=True)
    val_src = pad_sequence(process_sequences(val_df['latin'], input_tokenizer), batch_first=True)
    val_tgt = pad_sequence(process_sequences(val_df['devanagari'], output_tokenizer), batch_first=True)

    train_loader = DataLoader(TransliterationDataset(train_src, train_tgt), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TransliterationDataset(val_src, val_tgt), batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, input_tokenizer, output_tokenizer
