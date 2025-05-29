import numpy as np
import torch
import matplotlib.pyplot as plt
import minari
from model import DecisionTransformer
from config import MAX_LENGTH, device

def run_episode(model, state_mean, state_std, scale, target_return):
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    env = dataset.recover_environment(render_mode="rgb_array", eval_env=True)
    obs, _ = env.reset()

    act_dim = env.action_space.shape[0]
    state_hist = [obs.copy() for _ in range(MAX_LENGTH)]
    action_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist = list(range(MAX_LENGTH))

    total_reward = 0.0
    done = False

    while not done:
        states = np.stack(state_hist, 0).astype(np.float32)
        states = (states - state_mean) / state_std
        S = torch.from_numpy(states).to(device)
        A = torch.from_numpy(np.stack(action_hist, 0)).to(device)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(-1).to(device)
        Tm = torch.tensor(ts_hist, dtype=torch.long).to(device)

        with torch.no_grad():
            a = model.get_action(S, A, R, Tm)
        a_np = a.cpu().numpy().astype(np.float32)

        obs, rew, term, trunc, _ = env.step(a_np)
        total_reward += float(rew)
        done = term or trunc

        state_hist = state_hist[1:] + [obs.copy()]
        action_hist = action_hist[1:] + [a_np]
        new_rtg = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist = ts_hist[1:] + [(ts_hist[-1] + 1) % MAX_LENGTH]

    env.close()
    return total_reward

def main():
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    all_obs = np.concatenate([ep.observations[:-1] for ep in episodes], axis=0)
    state_mean = all_obs.mean(axis=0).astype(np.float32)
    state_std = all_obs.std(axis=0).astype(np.float32) + 1e-6
    all_rets = [float(ep.rewards.sum()) for ep in episodes]
    scale = max(all_rets)

    state_dim = episodes[0].observations.shape[1]
    act_dim = episodes[0].actions.shape[1]

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=MAX_LENGTH,
        transformer_name="gpt2",
        action_tanh=True,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
        scale=scale,
    ).to(device)

    ckpt = torch.load("dt_adroit_door_epoch10.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    rtg_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    rtg_values = [frac * scale for frac in rtg_fractions]
    avg_returns = []

    for rtg in rtg_values:
        print(f"Running episodes with RTG = {rtg:.2f}")
        returns = [run_episode(model, state_mean, state_std, scale, rtg) for _ in range(3)]
        avg_ret = np.mean(returns)
        avg_returns.append(avg_ret)
        print(f"Average return: {avg_ret:.2f}")

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(rtg_values, avg_returns, marker='o')
    plt.xlabel("Target Return (RTG)")
    plt.ylabel("Average Achieved Return")
    plt.title("Adroit Hand Door: Reward vs Target Return")
    plt.grid(True)
    plt.savefig("adroit_rtg_vs_return.png")
    print("Plot saved as adroit_rtg_vs_return.png")

if __name__ == "__main__":
    main()
