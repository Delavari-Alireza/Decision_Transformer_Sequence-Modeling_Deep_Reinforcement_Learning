from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn

class FullBlockVectorDataset(Dataset):
    def __init__(self, episodes, max_length):
        self.max_length = max_length
        self.episodes = [ep for ep in episodes if len(ep.actions) >= max_length]
    def __len__(self):
        return len(self.episodes)
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        obs  = np.array(ep.observations[:-1], dtype=np.float32)  # (N,39)
        acts = np.array(ep.actions,      dtype=np.float32)      # (N,28)
        rews = np.array(ep.rewards,      dtype=np.float32)      # (N,)
        N = acts.shape[0]; T=self.max_length
        start = np.random.randint(0, N-T+1)
        o = obs[start:start+T]
        a = acts[start:start+T]
        r = rews[start:start+T]
        # RTG
        rtg = np.zeros((T,1), dtype=np.float32)
        run=0
        for i in reversed(range(T)):
            run += r[i]; rtg[i]=run
        timesteps = np.arange(T, dtype=np.int64)
        mask = np.ones(T, dtype=np.float32)
        return {
            "states": torch.from_numpy(o),
            "actions": torch.from_numpy(a),
            "returns_to_go": torch.from_numpy(rtg),
            "timesteps": torch.from_numpy(timesteps),
            "attention_mask": torch.from_numpy(mask)
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}