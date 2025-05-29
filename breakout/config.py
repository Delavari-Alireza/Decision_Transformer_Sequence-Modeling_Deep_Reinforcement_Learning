
import torch

MAX_LENGTH = 20
BATCH_SIZE = 16
LR         = 1e-5
EPOCHS     = 100
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
