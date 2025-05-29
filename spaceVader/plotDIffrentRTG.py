import numpy as np
import torch
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import cv2

from model_cnn import DecisionTransformerCNN
from config import MAX_LENGTH, device


def run_episode(target_return, model, env, act_dim):
    obs, _ = env.reset()
    first_frame = (obs.astype(np.float32) / 255.0).transpose(2, 0, 1)
    state_hist = [first_frame.copy() for _ in range(MAX_LENGTH)]
    action_hist = [0 for _ in range(MAX_LENGTH)]
    ts_hist = list(range(MAX_LENGTH))
    returns_hist = [float(target_return)] * MAX_LENGTH

    total_reward = 0.0
    done = False

    while not done:
        S = torch.from_numpy(np.stack(state_hist, 0)).unsqueeze(0).to(device)
        A = torch.from_numpy(np.stack(action_hist, 0)).unsqueeze(0).to(device)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        Tm = torch.tensor(ts_hist, dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.ones(1, MAX_LENGTH, dtype=torch.bool, device=device)

        with torch.no_grad():
            _, logits, _ = model(S, A, R, Tm, mask)
        a = int(logits[0, -1].argmax().cpu().numpy())

        obs, rew, term, trunc, _ = env.step(a)
        total_reward += float(rew)
        done = term or trunc

        f = (obs.astype(np.float32) / 255.0).transpose(2, 0, 1)
        state_hist = state_hist[1:] + [f]
        action_hist = action_hist[1:] + [a]
        new_rtg = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_next = (ts_hist[-1] + 1) % MAX_LENGTH
        ts_hist = ts_hist[1:] + [ts_next]

    return total_reward


def main():
    # Setup
    scale = 600.0  # or use your precomputed scale
    env = gym.make(
        "SpaceInvaders-v4",
        render_mode="rgb_array",
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0,
        full_action_space=False,
        max_num_frames_per_episode=108000,
    )
    obs, _ = env.reset()
    C, H, W = obs.shape[2], obs.shape[0], obs.shape[1]
    act_dim = env.action_space.n

    model = DecisionTransformerCNN(
        state_shape=(C, H, W),
        act_dim=act_dim,
        max_length=MAX_LENGTH,
        transformer_name="gpt2",
        state_mean=None,
        state_std=None,
        scale=scale,
    ).to(device)

    ckpt = torch.load("dt_cnn_no_pad_epoch15.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Evaluate different RTGs
    rtg_values = list(range(0, int(scale) + 100, 100))
    rewards = []

    for rtg in rtg_values:
        print(f"Running episode with RTG = {rtg}")
        reward = run_episode(rtg, model, env, act_dim)
        print(f"→ Episode return: {reward:.2f}")
        rewards.append(reward)

    env.close()

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(rtg_values, rewards, marker='o')
    plt.xlabel("Target Return (RTG)")
    plt.ylabel("Achieved Reward")
    plt.title("Decision Transformer: Reward vs Target RTG")
    plt.grid(True)
    plt.savefig("dt_rtg_vs_reward_plot.png")
    print("Plot saved as dt_rtg_vs_reward_plot.png")


if __name__ == "__main__":
    main()
