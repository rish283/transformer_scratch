from torch.utils.data import Dataset
import os
import torch

class TinyShakespeareTokenizer:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.itos = {i: char for i, char in enumerate(self.vocab)}
        self.stoi = {char: i for i, char in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])
    
    def __len__(self):
        return self.vocab_size


class TinyShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)
        