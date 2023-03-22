from torch import Tensor
from typing import Callable

from config import *
from data import get_batch
from models import Transformer


def train_transformer(train_data: Tensor, vocab_size: int) -> torch.nn.Module:
    model = Transformer(vocab_size, num_blocks=NUM_BLOCKS)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for i in range(NUM_EPOCHS):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch {i + 1}/{NUM_EPOCHS}, loss: {loss.item()}')

    return model


def generate_text(model: torch.nn.Module, num_tokens: int, decoder: Callable) -> str:
    start_idx = torch.zeros(1, 1, dtype=torch.long, device=device)
    return decoder(model.generate(start_idx, num_tokens=num_tokens)[0].tolist())
