import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from model import DecisionTransformer
from config import MAX_LENGTH, state_mean, state_std, scale, all_rets


def run_episode(env, model, device, target_rtg, state_mean, state_std, scale, max_ep_len=1000):
    obs, _ = env.reset()

    state_hist = [(obs - state_mean) / state_std]
    action_hist = [np.zeros(env.action_space.shape[0], np.float32)]
    rtg_hist = [torch.tensor([[target_rtg / scale]], device=device, dtype=torch.float32)]
    ts_hist = [torch.tensor(0, device=device, dtype=torch.long)]

    total_r = 0.0
    for t in range(max_ep_len):
        states = torch.from_numpy(np.stack(state_hist)).to(device, torch.float32)
        actions = torch.from_numpy(np.stack(action_hist)).to(device, torch.float32)
        returns_t = torch.cat(rtg_hist, 0).to(device, torch.float32)
        timesteps = torch.stack(ts_hist, 0).to(device, torch.long)

        a_t = model.get_action(states, actions, returns_t, timesteps)
        a_np = a_t.detach().cpu().numpy()

        obs, r, terminated, truncated, _ = env.step(a_np)
        done = terminated or truncated
        total_r += r

        state_hist.append((obs - state_mean) / state_std)
        action_hist.append(a_np)
        rtg_hist.append(rtg_hist[-1] - (r / scale))
        ts_hist.append(torch.tensor(t + 1, device=device, dtype=torch.long))

        if done:
            break

    return total_r


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup environment (no rendering)
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")

    # Load model
    model = DecisionTransformer(
        state_dim=state_mean.shape[0],
        act_dim=env.action_space.shape[0],
        max_length=MAX_LENGTH,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
        scale=scale,
    ).to(device)

    model.load_state_dict(torch.load("dt_epoch100.pt", map_location=device))
    model.eval()

    # Try multiple RTG values
    rtg_values = list(range(0, int(max(all_rets) + 1), 250))
    episode_returns = []

    for rtg in rtg_values:
        print(f"Evaluating with RTG = {rtg}")
        ep_return = run_episode(env, model, device, rtg, state_mean, state_std, scale)
        print(f"→ Return = {ep_return:.2f}")
        episode_returns.append(ep_return)

    env.close()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(rtg_values, episode_returns, marker='o')
    plt.xlabel("Target Return (RTG)")
    plt.ylabel("Achieved Reward")
    plt.title("Decision Transformer: Reward vs RTG (HalfCheetah-v5)")
    plt.grid(True)
    plt.savefig("halfcheetah_rtg_vs_reward.png")
    print("Plot saved as 'halfcheetah_rtg_vs_reward.png'")
