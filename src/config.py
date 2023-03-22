import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EMBED = 32
BLOCK_SIZE = 8
NUM_HEADS = 4
