import torch.nn
from torch import Tensor
from torch.nn import functional as F

from typing import Optional, Tuple

from config import *


class Head(torch.nn.Module):

    def __init__(self, head_size: int):
        super().__init__()
        self.key = torch.nn.Linear(NUM_EMBED, head_size, bias=False)
        self.query = torch.nn.Linear(NUM_EMBED, head_size, bias=False)
        self.value = torch.nn.Linear(NUM_EMBED, head_size, bias=False)
        # triangular matrix for incremental averaging
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x: Tensor):
        b, t, c = x.shape
        key = self.key(x)
        query = self.query(x)
        wei = query @ key.transpose(-2, -1) * c**(-.5)  # scaling to prevent exploding softmax
        wei = torch.masked_fill(wei, self.tril[:t, :t] == 0, float('-inf'))  # incremental averaging using `tril`
        wei = F.softmax(wei, dim=1)  # softmax to convert to probabilities
        wei = self.dropout(wei)

        return wei @ self.value(x)


class Multihead(torch.nn.Module):

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(NUM_EMBED, NUM_EMBED)  # projection layer
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(NUM_EMBED, 4*NUM_EMBED),
            torch.nn.ReLU(),
            torch.nn.Linear(4*NUM_EMBED, NUM_EMBED),
            torch.nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.mlp(x)


class Block(torch.nn.Module):

    def __init__(self, head_size: int):
        super().__init__()
        self.self_attention = Multihead(NUM_HEADS, head_size=head_size // NUM_HEADS)
        self.mlp = FeedForward()
        self.layer_norm_1 = torch.nn.LayerNorm(NUM_EMBED)
        self.layer_norm_2 = torch.nn.LayerNorm(NUM_EMBED)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.self_attention(x)  # residual connection
        x = self.layer_norm_2(x)
        x = x + self.mlp(x)
        return x


class Transformer(torch.nn.Module):

    def __init__(self, vocab_size: int, num_blocks: int):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, NUM_EMBED)
        self.positional_embedding_table = torch.nn.Embedding(NUM_EMBED, NUM_EMBED)
        self.blocks = torch.nn.Sequential(*[Block(head_size=NUM_EMBED) for _ in range(num_blocks)])
        self.layer_norm = torch.nn.LayerNorm(NUM_EMBED)
        self.lm_head = torch.nn.Linear(NUM_EMBED, vocab_size)

    def forward(self, idx: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        b, t = idx.shape  # (batch_size, block_size)

        # get token & positional embeddings and sum them
        token_embed = self.token_embedding_table(idx)  # (batch_size, block_size, num_embed)
        posit_embed = self.positional_embedding_table(torch.arange(t, device=device))  # (block_size, num_embed)
        x = token_embed + posit_embed  # (batch_size, block_size, vocab_size)

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)   # pass to decoder language model head

        logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])

        if not torch.is_tensor(target):
            loss = None
        else:
            target = target.view(target.shape[0] * target.shape[1], )
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx: Tensor, num_tokens: int) -> Tensor:

        for _ in range(num_tokens):
            # crop the context that is fed to network to `block_size`
            # avoids overflow during positional embedding (as table size is `block_size`)
            context = idx[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            logits = logits[-1, :]
            prob = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, num_samples=1)
            next_idx = next_idx.view(1, 1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
