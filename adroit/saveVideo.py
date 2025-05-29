import os
import numpy as np
import torch
import minari
import gymnasium as gym
import cv2

from model import DecisionTransformer
from config import MAX_LENGTH, device

def record_episode(model, state_mean, state_std, scale, target_return, video_path="adroit_replay.mp4", fps=25):
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    env = dataset.recover_environment(render_mode="rgb_array", eval_env=True)
    obs, _ = env.reset()

    act_dim = env.action_space.shape[0]
    state_hist = [obs.copy() for _ in range(MAX_LENGTH)]
    action_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist = list(range(MAX_LENGTH))

    frame0 = env.render()
    H, W, _ = frame0.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

    total_reward = 0.0
    done = False
    while not done:
        states = np.stack(state_hist, 0).astype(np.float32)
        states = (states - state_mean) / state_std
        S = torch.from_numpy(states).to(device)
        A = torch.from_numpy(np.stack(action_hist,0)).to(device)
        R = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(-1).to(device)
        Tm = torch.tensor(ts_hist, dtype=torch.long).to(device)

        with torch.no_grad():
            a = model.get_action(S, A, R, Tm)
        a_np = a.cpu().numpy().astype(np.float32)

        obs, rew, term, trunc, _ = env.step(a_np)
        total_reward += float(rew)
        done = term or trunc

        frame = env.render()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

        state_hist = state_hist[1:] + [obs.copy()]
        action_hist = action_hist[1:] + [a_np]
        new_rtg = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist = ts_hist[1:] + [(ts_hist[-1] + 1) % MAX_LENGTH]

    writer.release()
    cv2.destroyAllWindows()
    env.close()
    return total_reward


if __name__ == "__main__":
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    all_obs = np.concatenate([ep.observations[:-1] for ep in episodes], axis=0)
    state_mean = all_obs.mean(axis=0).astype(np.float32)
    state_std  = all_obs.std(axis=0).astype(np.float32) + 1e-6
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

    target_return = 0  # or scale, or any target RTG you want
    out_video = "adroit_replay.mp4"
    if os.path.exists(out_video):
        os.remove(out_video)

    ret = record_episode(model, state_mean, state_std, scale, target_return,
                         video_path=out_video, fps=25)
    print(f"Episode return: {ret:.2f} — saved replay to {out_video}")
