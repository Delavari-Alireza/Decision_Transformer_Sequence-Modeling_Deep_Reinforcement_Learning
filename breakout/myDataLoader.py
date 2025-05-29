
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class FullBlockDataset(Dataset):
    def __init__(self, dataset, max_length):
        self.max_length = max_length
        self.episodes = []
        for ep in dataset.iterate_episodes():
            if len(ep.actions) >= max_length:
                self.episodes.append(ep)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        obs  = ep.observations.astype(np.float32) / 255.0    # (N+1,H,W,3)
        acts = ep.actions.astype(np.int64)                  # (N,)
        rews = ep.rewards.astype(np.float32)                # (N,)
        N = acts.shape[0]
        T = self.max_length
        start = np.random.randint(0, N - T + 1)
        o = obs[start:start+T]
        a = acts[start:start+T]
        r = rews[start:start+T]
        rtg = np.zeros(T, dtype=np.float32)
        running = 0.0
        for i in reversed(range(T)):
            running += r[i]
            rtg[i] = running

        timesteps = np.arange(T, dtype=np.int64)
        o = o.transpose(0, 3, 1, 2)

        return {
            'states': torch.from_numpy(o),                  # (T, 3, H, W)
            'actions': torch.from_numpy(a),                 # (T,)
            'returns_to_go': torch.from_numpy(rtg).unsqueeze(-1),  # (T, 1)
            'timesteps': torch.from_numpy(timesteps)        # (T,)
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0)
            for k in batch[0]}
