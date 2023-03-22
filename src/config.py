import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EMBED = 384  # must be divisible by `NUM_HEADS`
BLOCK_SIZE = 256
NUM_HEADS = 6
DROPOUT = 0.2
NUM_BLOCKS = 6
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5000
BATCH_SIZE = 64
