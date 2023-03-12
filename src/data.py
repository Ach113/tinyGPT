import torch
from torch import Tensor
from typing import List, Tuple


class Dataset:

    def __init__(self, file_name: str):
        with open(file_name, 'r') as f:
            text = f.read()

        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.atoi = {ch: self.vocab.index(ch) for ch in self.vocab}
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, s: str) -> List[int]:
        return [self.atoi[ch] for ch in s]

    def decode(self, m: List[int]) -> str:
        return ''.join([self.vocab[i] for i in m])


def get_batch(data, batch_size: int, block_size: int) -> Tuple[Tensor, Tensor]:
    indices = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
    return x, y
