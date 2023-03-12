import torch
from torch import Tensor
from torch.nn import functional as F

from typing import Optional, Tuple


class BigramModel(torch.nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        logits = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
        if not torch.is_tensor(target):
            loss = None
        else:
            target = target.view(target.shape[0] * target.shape[1], )
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx: Tensor, num_tokens: int) -> Tensor:

        for _ in range(num_tokens):
            logits, _ = self(idx)
            logits = logits[-1, :]
            prob = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, num_samples=1)
            next_idx = next_idx.view(1, 1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
