import numpy as np
import torch
import gymnasium as gym
import minari
from model_cnn import DecisionTransformerCNN
from config import MAX_LENGTH, device
import ale_py

device = "cpu"  # or "cuda" if available
MAX_LENGTH = 15


def run_episode(model, target_return, scale):
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0

    state_hist   = [obs for _ in range(MAX_LENGTH)]
    action_hist  = [0 for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist      = list(range(MAX_LENGTH))

    total_reward = 0.0
    done = False

    while not done:
        frames = np.stack(state_hist).transpose(0, 3, 1, 2)  # (T, C, H, W)
        S = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
        A = torch.tensor(action_hist, dtype=torch.long).unsqueeze(0).to(device)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        Tm = torch.tensor(ts_hist, dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.ones(1, MAX_LENGTH, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, _ = model(S, A, R, Tm, mask)
            a = int(logits[0, -1].argmax().item())

        new_obs, rew, term, trunc, _ = env.step(a)
        new_obs = new_obs.astype(np.float32) / 255.0
        total_reward += float(rew)
        done = term or trunc

        state_hist   = state_hist[1:] + [new_obs]
        action_hist  = action_hist[1:] + [a]
        new_rtg      = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist      = ts_hist[1:] + [(ts_hist[-1] + 1) % MAX_LENGTH]

    env.close()
    return total_reward


def main():
    # Load dataset & compute RTG scale
    dataset = minari.load_dataset("atari/breakout/expert-v0")
    scale = max(float(ep.rewards.sum()) for ep in dataset.iterate_episodes()
                if len(ep.actions) >= MAX_LENGTH)
    print(f"RTG scale = {scale:.1f}")

    # Setup model
    spec = dataset.spec
    C, H, W = spec.observation_space.shape[2], spec.observation_space.shape[0], spec.observation_space.shape[1]
    act_dim = spec.action_space.n

    model = DecisionTransformerCNN(
        state_shape=(C, H, W),
        act_dim=act_dim,
        max_length=MAX_LENGTH,
        transformer_name="gpt2",
        scale=scale
    ).to(device)

    model.load_state_dict(torch.load("dt_cnn_no_pad_epoch_cpu_dis100.pt", map_location=device))
    model.eval()

    # Sweep across RTGs
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tgt = frac * scale
        rets = [run_episode(model, tgt, scale) for _ in range(5)]
        print(f"RTG = {tgt:6.1f} → Avg Return = {np.mean(rets):.1f} over {len(rets)} episodes")

if __name__ == "__main__":
    main()
