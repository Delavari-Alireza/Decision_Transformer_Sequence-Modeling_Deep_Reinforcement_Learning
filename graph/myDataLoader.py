import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GraphTrajectoryDataset(Dataset):
    def __init__(self, episodes, graph, max_length):

        self.episodes   = episodes
        self.G          = graph
        self.max_length = max_length
        self.num_nodes  = graph.number_of_nodes()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep    = self.episodes[idx]
        S_all = ep["states"]
        A_all = ep["actions"]
        R_all = ep["rewards"]
        L_all = len(S_all)
        T     = self.max_length

        # 1) sample or pad to length T
        if L_all >= T:
            start = np.random.randint(0, L_all - T + 1)
            S = S_all[start:start+T]
            A = A_all[start:start+T]
            R = R_all[start:start+T]
            m = [1]*T
        else:
            pad = T - L_all
            S = [0]*pad + S_all
            A = [0]*pad + A_all
            R = [0.0]*pad + R_all
            m = [0]*pad + [1]*L_all

        # 2) compute RTG safely (no negative strides)
        arr = np.array(R, dtype=np.float32)
        rtg = np.cumsum(arr[::-1], axis=0)[::-1].copy().reshape(T,1)

        # 3) build valid_mask per timestep (never allow “stay”)
        vm = np.zeros((T, self.num_nodes), dtype=bool)
        for t, s in enumerate(S):
            for nb in self.G.neighbors(int(s)):
                if nb != s:
                    vm[t, nb] = True

        return {
            "states":         torch.tensor(S,   dtype=torch.long),
            "actions":        torch.tensor(A,   dtype=torch.long),
            "returns_to_go":  torch.tensor(rtg, dtype=torch.float32),
            "timesteps":      torch.arange(T,   dtype=torch.long),
            "attention_mask": torch.tensor(m,   dtype=torch.float32),
            "valid_mask":     torch.tensor(vm,  dtype=torch.bool),
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0)
            for k in batch[0]}
