#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

# -------------------------------
# 1. Dummy Dataset
# -------------------------------
data = [
    ("namaste", "नमस्ते"),
    ("dilli",   "दिल्ली"),
    ("pyaar",   "प्यार"),
    ("hindi",   "हिंदी"),
    ("shukriya","शुक्रिया"),
]

def tokenize_data(pairs):
    # src as list(chars), tgt wrapped with <sos>/<eos>
    return [(list(x), ['<sos>'] + list(y) + ['<eos>']) for x, y in pairs]

data = tokenize_data(data)

# -------------------------------
# 2. Vocabulary
# -------------------------------
def build_vocab(seqs):
    # flatten and sort unique chars
    chars = sorted({ch for seq in seqs for ch in seq})
    vocab = {ch: idx for idx, ch in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    return vocab

src_vocab = build_vocab([x for x, _ in data])
tgt_vocab = build_vocab([y for _, y in data])
src_ivocab = {i: c for c, i in src_vocab.items()}
tgt_ivocab = {i: c for c, i in tgt_vocab.items()}

# -------------------------------
# 3. Dataset
# -------------------------------
class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_src_len=10, max_tgt_len=12):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        # map to ids
        src_ids = [self.src_vocab.get(ch, self.src_vocab['<pad>']) for ch in src]
        tgt_ids = [self.tgt_vocab.get(ch, self.tgt_vocab['<pad>']) for ch in tgt]

        # pad/truncate
        src_ids = src_ids[:self.max_src_len] + [self.src_vocab['<pad>']] * max(0, self.max_src_len - len(src_ids))
        tgt_ids = tgt_ids[:self.max_tgt_len] + [self.tgt_vocab['<pad>']] * max(0, self.max_tgt_len - len(tgt_ids))

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# -------------------------------
# 4. Models
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, cell_type='GRU'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cls = getattr(nn, cell_type.upper())
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, src):
        emb = self.embedding(src)
        outputs, hidden = self.rnn(emb)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, cell_type='GRU'):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = getattr(nn, cell_type.upper())
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, inp, hidden):
        # inp: (batch,) token ids
        inp = inp.unsqueeze(1)                     # (batch,1)
        emb = self.embedding(inp)                  # (batch,1,emb_dim)
        out, hidden = self.rnn(emb, hidden)        # out:(batch,1,hid), hidden:...
        preds = self.fc(out.squeeze(1))            # (batch, output_dim)
        return preds, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing=0.75):
        batch_size, max_len = src.size(0), tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_len, vocab_size, device=src.device)
        hidden = self.encoder(src)

        # first decoder input = <sos> from tgt
        input_tok = tgt[:, 0]

        for t in range(1, max_len):
            preds, hidden = self.decoder(input_tok, hidden)
            outputs[:, t] = preds

            teacher = torch.rand(1).item() < teacher_forcing
            input_tok = tgt[:, t] if teacher else preds.argmax(1)

        return outputs

# -------------------------------
# 5. Training & Inference
# -------------------------------
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out = model(src, tgt)
        # shift out / tgt to ignore <sos>
        out_flat = out[:,1:,:].reshape(-1, out.size(-1))
        tgt_flat = tgt[:,1:].reshape(-1)
        loss = crit(out_flat, tgt_flat)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, word, device, max_len=12):
    model.eval()
    with torch.no_grad():
        src_ids = [src_vocab.get(ch, src_vocab['<pad>']) for ch in word]
        src_ids = src_ids[:10] + [src_vocab['<pad>']] * max(0, 10-len(src_ids))
        src = torch.tensor(src_ids, device=device).unsqueeze(0)

        hidden = model.encoder(src)
        inp = torch.tensor([tgt_vocab['<sos>']], device=device)
        result = []

        for _ in range(max_len):
            preds, hidden = model.decoder(inp, hidden)
            top = preds.argmax(1).item()
            ch = tgt_ivocab[top]
            if ch == '<eos>': break
            result.append(ch)
            inp = torch.tensor([top], device=device)

    return "".join(result)

# -------------------------------
# 6. Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim',    type=int, default=32)
    parser.add_argument('--hid_dim',    type=int, default=64)
    parser.add_argument('--enc_layers', type=int, default=1)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--cell_type',  type=str, default='GRU', choices=['RNN','LSTM','GRU'])
    parser.add_argument('--epochs',     type=int, default=20)
    args, _ = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset + Loader
    ds = TransliterationDataset(data, src_vocab, tgt_vocab)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # Model
    enc = Encoder(len(src_vocab), args.emb_dim, args.hid_dim, args.enc_layers, args.cell_type)
    dec = Decoder(len(tgt_vocab), args.emb_dim, args.hid_dim, args.dec_layers, args.cell_type)
    model = Seq2Seq(enc, dec).to(device)

    opt = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training
    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, loader, opt, crit, device)
        print(f"Epoch {ep}/{args.epochs}   Loss: {loss:.4f}")

    # Inference demo
        # Inference demo
    print("\n--- Predictions ---")
    for w_list, _ in data:
        w = ''.join(w_list)                  # join list of chars into string
        pred = predict(model, w, device)
        print(f"{w:10s} -> {pred}")

if __name__ == '__main__':
    main()

