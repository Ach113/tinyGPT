import torch
from torch import Tensor
from typing import Callable

from models import BigramModel
from data import get_batch


def train_bigram(train_data: Tensor, vocab_size: int, batch_size: int, block_size: int,
                 epochs: int, lr: float = 1e-3) -> torch.nn.Module:
    model = BigramModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(epochs):
        x, y = get_batch(train_data, batch_size, block_size)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch {i + 1}/{epochs}, loss: {loss.item()}')

    return model


def generate_text(model: torch.nn.Module, num_tokens: int, decoder: Callable) -> str:
    start_idx = torch.zeros(1, 1, dtype=torch.long)
    return decoder(model.generate(start_idx, num_tokens=num_tokens)[0].tolist())
