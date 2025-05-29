import torch
import gymnasium as gym
import numpy as np
from model import DecisionTransformer
from config import MAX_LENGTH, state_mean, state_std, scale, all_rets

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) env
    env = gym.make("HalfCheetah-v5", render_mode="human")
    obs, _ = env.reset()

    # 2) build & load
    model = DecisionTransformer(
        state_dim  = state_mean.shape[0],
        act_dim    = env.action_space.shape[0],
        max_length = MAX_LENGTH,
        state_mean = torch.from_numpy(state_mean),
        state_std  = torch.from_numpy(state_std),
        scale      = scale,
    ).to(device)

    model.load_state_dict(torch.load("dt_epoch100.pt", map_location=device))
    model.eval()

    # 3) histories
    state_hist  = [ (obs - state_mean) / state_std ]                  # list of np arrays
    action_hist = [ np.zeros(env.action_space.shape[0], np.float32) ]
    rtg_hist    = [ torch.tensor([[ max(all_rets) / scale ]],
                                 device=device, dtype=torch.float32) ]
    ts_hist     = [ torch.tensor(0, device=device, dtype=torch.long) ]  # scalar

    total_r = 0.0
    for t in range(1000):
        # stack histories
        states     = torch.from_numpy(np.stack(state_hist, 0)).to(device, torch.float32)
        actions    = torch.from_numpy(np.stack(action_hist,0)).to(device, torch.float32)
        returns_t  = torch.cat(rtg_hist, 0).to(device, torch.float32)
        timesteps  = torch.stack(ts_hist, 0).to(device, torch.long)

        # get action
        a_t = model.get_action(states, actions, returns_t, timesteps)
        a_np = a_t.detach().cpu().numpy()

        # step env
        obs, r, terminated, truncated, _ = env.step(a_np)
        done = terminated or truncated
        total_r += r

        # update histories
        state_hist.append((obs - state_mean) / state_std)
        action_hist.append(a_np)
        rtg_hist.append(rtg_hist[-1] - (r / scale))
        ts_hist.append(torch.tensor(t+1, device=device, dtype=torch.long))

        env.render()
        if done:
            break

    print("Episode return:", total_r)
    env.close()
