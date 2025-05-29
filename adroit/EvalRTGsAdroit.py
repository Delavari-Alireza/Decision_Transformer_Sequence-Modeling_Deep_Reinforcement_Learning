import numpy as np
import torch
import minari
import gymnasium as gym
import ale_py                            # register ALE backend
import gymnasium_robotics               # register robotics envs

from model import DecisionTransformer
from config import MAX_LENGTH, device

def run_episode(model, target_return):
    # 1) recover the exact AdroitHandDoor-v1 env
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    env     = dataset.recover_environment(eval_env=True)
    obs, _  = env.reset()  # obs: (39,)

    # 2) prepare rolling buffers (no zero‐padding!)
    act_dim = env.action_space.shape[0]
    state_hist   = [obs.copy() for _ in range(MAX_LENGTH)]    # list of (39,)
    action_hist  = [np.zeros(act_dim, dtype=np.float32)
                    for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist      = list(range(MAX_LENGTH))

    total_reward = 0.0
    done = False

    while not done:
        # 3) build unbatched inputs
        states = torch.from_numpy(np.stack(state_hist,0).astype(np.float32)) \
                        .to(device)                       # (T,39)
        actions= torch.from_numpy(np.stack(action_hist,0)) \
                        .to(device)                       # (T,28)
        rtgs   = torch.tensor(returns_hist, dtype=torch.float32) \
                        .unsqueeze(-1).to(device)         # (T,1)
        timesteps = torch.tensor(ts_hist, dtype=torch.long) \
                        .to(device)                       # (T,)

        # 4) get_action will add batch dim internally
        with torch.no_grad():
            a = model.get_action(states, actions, rtgs, timesteps)  # (act_dim,)

        a_np = a.cpu().numpy().astype(np.float32)

        # 5) step the environment
        obs, rew, term, trunc, _ = env.step(a_np)
        total_reward += float(rew)
        done = term or trunc

        # 6) update rolling buffers
        state_hist  = state_hist[1:]   + [obs.copy()]
        action_hist = action_hist[1:]  + [a_np]
        # subtract raw reward from last RTG
        new_rtg     = returns_hist[-1] - float(rew)
        returns_hist= returns_hist[1:] + [new_rtg]
        ts_hist     = ts_hist[1:]      + [(ts_hist[-1]+1) % MAX_LENGTH]

    env.close()
    return total_reward

def main():
    # recompute normalization & RTG scale on all episodes
    dataset  = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    all_obs  = np.concatenate([ep.observations[:-1] for ep in episodes], axis=0)
    mean     = all_obs.mean(axis=0)
    std      = all_obs.std(axis=0) + 1e-6
    all_rets = [float(ep.rewards.sum()) for ep in episodes]
    scale    = max(all_rets)
    print(f"Computed state mean/std and scale = {scale:.3f}")

    # build & load the model
    state_dim = episodes[0].observations.shape[1]
    act_dim   = episodes[0].actions.shape[1]
    model = DecisionTransformer(
        state_dim  = state_dim,
        act_dim    = act_dim,
        max_length = MAX_LENGTH,
        transformer_name = "gpt2",
        action_tanh = True,
        state_mean  = torch.from_numpy(mean),
        state_std   = torch.from_numpy(std),
        scale       = scale,
    ).to(device)

    ckpt = torch.load("dt_adroit_door_epoch10.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # sweep target returns (fractions of expert max)
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        tgt = frac * scale
        returns = [run_episode(model, tgt) for _ in range(5)]
        print(f"target_return={tgt:.1f} -> avg_return={np.mean(returns):.1f}")

if __name__ == "__main__":
    main()
