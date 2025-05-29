import numpy as np
import torch
import minari
import gymnasium as gym
import ale_py                   # ensure ALE backend
import gymnasium_robotics      # register robotics envs

from model_cnn import DecisionTransformerRC
from config import MAX_LENGTH, device

def run_episode(model, target_return, state_mean, state_std, scale):
    # 1) recover the exact eval environment
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    env     = dataset.recover_environment(eval_env=True)
    obs, _  = env.reset()  # obs: (39,)

    # 2) prepare fixed‐length history
    act_dim = env.action_space.shape[0]
    state_hist   = [obs.copy() for _ in range(MAX_LENGTH)]
    action_hist  = [np.zeros(act_dim, dtype=np.float32) for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist      = list(range(MAX_LENGTH))

    total_reward = 0.0
    done = False

    while not done:
        # build inputs (ensure float32 for states)
        S_np = np.stack(state_hist, 0).astype(np.float32)  # (T,39), float32
        S = torch.from_numpy(S_np).unsqueeze(0).to(device) # (1,T,39)
        A = torch.from_numpy(np.stack(action_hist,0)).unsqueeze(0).to(device)  # (1,T,28)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1,T,1)
        Tm= torch.tensor(ts_hist,     dtype=torch.long).unsqueeze(0).to(device)                   # (1,T)

        # normalize states
        S = (S - state_mean.to(device)) / state_std.to(device)

        # get action (unbatched)
        with torch.no_grad():
            a = model.get_action(S[0], A[0], R[0], Tm[0])  # returns (28,)
        a_np = a.cpu().numpy().astype(np.float32)

        # step
        obs, rew, term, trunc, _ = env.step(a_np)
        total_reward += float(rew)
        done = term or trunc

        # update buffers
        state_hist   = state_hist[1:]   + [obs.copy()]
        action_hist  = action_hist[1:]  + [a_np]
        new_rtg      = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist      = ts_hist[1:]      + [(ts_hist[-1] + 1) % MAX_LENGTH]

    env.close()
    return total_reward

def main():
    # recompute normalization & scale
    dataset  = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    all_obs  = np.concatenate([ep.observations[:-1] for ep in episodes], axis=0).astype(np.float32)
    mean     = torch.from_numpy(all_obs.mean(axis=0)).float()
    std      = torch.from_numpy(all_obs.std(axis=0) + 1e-6).float()
    all_rets = [float(ep.rewards.sum()) for ep in episodes]
    scale    = max(all_rets)
    print(f"Computed normalization and scale = {scale:.3f}")

    # build & load model
    state_dim = episodes[0].observations.shape[1]  # 39
    act_dim   = episodes[0].actions.shape[1]       # 28
    model = DecisionTransformerRC(
        state_dim  = state_dim,
        act_dim    = act_dim,
        max_length = MAX_LENGTH,
        transformer_name="gpt2",
        action_tanh=True,
        state_mean = mean,
        state_std  = std,
        scale      = scale
    ).to(device)
    ckpt = torch.load("dt_adroit_rc_epoch5.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # sweep RTG fractions
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tgt = frac * scale
        rets = [run_episode(model, tgt, mean, std, scale) for _ in range(5)]
        print(f"RTG={tgt:6.1f} -> avg_return={np.mean(rets):.1f} over {len(rets)} eps")

if __name__ == "__main__":
    main()
