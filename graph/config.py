
import torch


device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 20
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE  = "graph_random_1000.pkl"   # your 1000-episode pickle


