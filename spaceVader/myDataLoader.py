from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn

class FullBlockDataset(Dataset):
    def __init__(self, episodes, max_length):
        self.max_length = max_length
        # keep only episodes long enough
        self.episodes = [ep for ep in episodes if len(ep.actions) >= max_length]
    def __len__(self):
        return len(self.episodes)
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        obs  = ep.observations.astype(np.float32) / 255.0    # (N+1,H,W,3)
        acts = ep.actions.astype(np.int64)                  # (N,)
        rews = ep.rewards.astype(np.float32)                # (N,)
        N = acts.shape[0]
        T = self.max_length
        # sample full-length block
        start = np.random.randint(0, N - T + 1)
        o = obs[start:start+T]
        a = acts[start:start+T]
        r = rews[start:start+T]
        # compute RTG
        rtg = np.zeros(T, dtype=np.float32)
        running = 0.0
        for i in reversed(range(T)):
            running += r[i]
            rtg[i] = running
        timesteps = np.arange(T, dtype=np.int64)
        # transpose
        o = o.transpose(0,3,1,2)
        return {
            'states': torch.from_numpy(o),           # (T,3,H,W)
            'actions': torch.from_numpy(a),          # (T,)
            'returns_to_go': torch.from_numpy(rtg).unsqueeze(-1), # (T,1)
            'timesteps': torch.from_numpy(timesteps) # (T,)
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0)
            for k in batch[0]}