
from torch.utils.data import Dataset
import numpy as np
import torch
from config import state_mean, state_std, scale

# ensure your means/stds are float32
state_mean = state_mean.astype(np.float32)
state_std  = state_std.astype(np.float32)

class TrajectoryDataset(Dataset):
    def __init__(self, hf_dataset, max_length):
        self.episodes   = hf_dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]

        # --- load and normalize states & rewards ---
        # obs: (N, S), originally float32
        obs  = np.array(ep["observations"], dtype=np.float32)
        # subtract & divide by float32 means/stds → still float32
        obs  = (obs - state_mean) / state_std

        acts = np.array(ep["actions"], dtype=np.float32)  # (N, A)
        rews = np.array(ep["rewards"], dtype=np.float32)  # (N,)
        rews = rews / scale                               # still float32

        N, T = acts.shape[0], self.max_length

        # --- sample or pad to length T ---
        if N >= T:
            start = np.random.randint(0, N - T + 1)
            o = obs[start:start+T]
            a = acts[start:start+T]
            r = rews[start:start+T]
            mask = np.ones(T, dtype=np.float32)
        else:
            pad = T - N
            o = np.vstack([np.zeros((pad, obs.shape[1]), dtype=np.float32), obs])
            a = np.vstack([np.zeros((pad, acts.shape[1]), dtype=np.float32), acts])
            r = np.hstack([np.zeros(pad, dtype=np.float32), rews])
            mask = np.hstack([np.zeros(pad, dtype=np.float32), np.ones(N, dtype=np.float32)])

        # --- relative timesteps 0..T-1 ---
        timesteps = np.arange(T, dtype=np.int64)

        # --- returns-to-go on scaled rewards ---
        rtg = np.zeros(T, dtype=np.float32)
        running = 0.0
        for i in reversed(range(T)):
            running += r[i]
            rtg[i] = running

        return {
            "states":         torch.from_numpy(o),             # (T, S), float32
            "actions":        torch.from_numpy(a),             # (T, A), float32
            "returns_to_go":  torch.from_numpy(rtg).unsqueeze(-1),  # (T,1), float32
            "timesteps":      torch.from_numpy(timesteps),     # (T,),      int64
            "attention_mask": torch.from_numpy(mask),          # (T,),      float32
        }

def collate_fn(batch):
    return {k: torch.stack([d[k] for d in batch], dim=0) for k in batch[0]}
