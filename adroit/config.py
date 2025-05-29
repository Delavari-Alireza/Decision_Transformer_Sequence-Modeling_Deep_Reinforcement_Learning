
import torch

# — hyper-parameters —
MAX_LENGTH = 20
BATCH_SIZE = 16
LR         = 3e-4
EPOCHS     = 100
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


