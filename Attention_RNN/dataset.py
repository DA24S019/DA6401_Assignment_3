from torch.utils.data import Dataset

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
