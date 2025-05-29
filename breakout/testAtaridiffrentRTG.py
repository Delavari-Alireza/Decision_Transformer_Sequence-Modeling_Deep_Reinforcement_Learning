import numpy as np
import torch
import minari
import gymnasium as gym
import ale_py
import gymnasium_robotics

from model_cnn import DecisionTransformerCNN
from myDataLoader import FullBlockDataset, collate_fn
from config import MAX_LENGTH, device

device = "cpu"  # e.g. "cpu" or "cuda"
MAX_LENGTH = 15


def run_episode(model, target_return):
    # 1) make the env
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    obs, _ = env.reset()  # obs: (210, 160, 3), uint8

    # preprocess to float32 [0,1]
    obs = obs.astype(np.float32) / 255.0

    # 2) initialize history buffers
    #   store frames as HxWx3 float arrays
    state_hist   = [obs for _ in range(MAX_LENGTH)]
    # store actions as ints
    action_hist  = [0 for _ in range(MAX_LENGTH)]
    # RTG pad with the target
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist      = list(range(MAX_LENGTH))

    total_reward = 0.0
    done = False

    while not done:
        # --- build model input batch (1,L,3,H,W) ---
        # (T, H, W, C) → (T, C, H, W)
        frames = np.stack(state_hist, axis=0).transpose(0, 3, 1, 2)
        S = torch.from_numpy(frames).unsqueeze(0).to(device)              # (1,T,3,H,W)
        A = torch.tensor(action_hist, dtype=torch.long).unsqueeze(0).to(device)  # (1,T)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1,T,1)
        Tm= torch.tensor(ts_hist, dtype=torch.long).unsqueeze(0).to(device)            # (1,T)
        mask = torch.ones(1, MAX_LENGTH, dtype=torch.bool, device=device)

        # --- forward & pick action ---
        with torch.no_grad():
            logits, _ = model(S, A, R, Tm, mask)   # (1,T,act_dim), (1,T,1)
            a = int(logits[0, -1].argmax().item())

        # --- step env ---
        new_obs, rew, done, trunc, _ = env.step(a)
        new_obs = new_obs.astype(np.float32) / 255.0
        total_reward += float(rew)
        done = done or trunc

        # --- update histories ---
        state_hist   = state_hist[1:]   + [new_obs]
        action_hist  = action_hist[1:]  + [a]
        new_rtg      = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist      = ts_hist[1:]      + [(ts_hist[-1] + 1) % MAX_LENGTH]

    env.close()
    return total_reward

def main():
    # 1) load once
    dataset = minari.load_dataset("atari/breakout/expert-v0")
    scale = 0.0
    for ep in dataset.iterate_episodes():
        if len(ep.actions) >= MAX_LENGTH:
            ep_return = float(ep.rewards.sum())
            if ep_return > scale:
                scale = ep_return

    print(f" RTG scale={scale:.1f}")

    # 3) build model & load weights
    spec = dataset.spec
    C,H,W = spec.observation_space.shape[2], spec.observation_space.shape[0], spec.observation_space.shape[1]
    act_dim = spec.action_space.n

    model = DecisionTransformerCNN(
        state_shape=(C,H,W), act_dim=act_dim,
        max_length=MAX_LENGTH, transformer_name="gpt2",
        scale=scale
    ).to(device)
    ckpt = torch.load("dt_cnn_no_pad_epoch_cpu_dis100.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 4) sweep different target RTGs
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tgt = frac * scale
        rets = [run_episode(model, tgt) for _ in range(5)]
        print(f"RTG={tgt:6.1f} → avg_return={np.mean(rets):.1f} over {len(rets)} eps")

if __name__ == "__main__":
    main()
