import os
import numpy as np
import torch
import minari
import gymnasium as gym
import ale_py
import gymnasium_robotics
import cv2

from model import DecisionTransformer
from config import MAX_LENGTH, device


def record_episode(model, state_mean, state_std, scale, target_return, video_path="adroit_replay.mp4", fps=25):
    """
    Run one episode in AdroitHandDoor-v1 with DecisionTransformer,
    render on-screen, and save to video_path (mp4).
    Returns total episode reward.
    """
    # 1) recover environment with rgb_array
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    env = dataset.recover_environment(render_mode="rgb_array", eval_env=True)
    obs, _ = env.reset()  # obs: (39,)

    # 2) prepare rolling buffers
    act_dim = env.action_space.shape[0]
    state_hist   = [obs.copy() for _ in range(MAX_LENGTH)]
    action_hist  = [np.zeros(act_dim, dtype=np.float32) for _ in range(MAX_LENGTH)]
    returns_hist = [float(target_return)] * MAX_LENGTH
    ts_hist      = list(range(MAX_LENGTH))

    # 3) set up video writer
    frame0 = env.render()  # rgb array shape (H,W,3)
    H, W, _ = frame0.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

    total_reward = 0.0
    done = False

    cv2.namedWindow("Adroit DT", cv2.WINDOW_NORMAL)
    while not done:
        # normalize states
        states = np.stack(state_hist, 0).astype(np.float32)
        states = (states - state_mean) / state_std
        # build inputs
        S   = torch.from_numpy(states).to(device)                    # (T,39)
        A   = torch.from_numpy(np.stack(action_hist,0)).to(device)   # (T,28)
        R   = torch.tensor(returns_hist, dtype=torch.float32).unsqueeze(-1).to(device)  # (T,1)
        Tm  = torch.tensor(ts_hist, dtype=torch.long).to(device)      # (T,)

        # get action
        with torch.no_grad():
            a = model.get_action(S, A, R, Tm)                        # (28,)
        a_np = a.cpu().numpy().astype(np.float32)

        # step
        obs, rew, term, trunc, _ = env.step(a_np)
        total_reward += float(rew)
        done = term or trunc

        # render & record
        frame = env.render()  # rgb
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        cv2.imshow("Adroit DT", bgr)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

        # update buffers
        state_hist   = state_hist[1:]   + [obs.copy()]
        action_hist  = action_hist[1:]  + [a_np]
        new_rtg      = returns_hist[-1] - float(rew)
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist      = ts_hist[1:]      + [(ts_hist[-1] + 1) % MAX_LENGTH]

    # cleanup
    writer.release()
    cv2.destroyAllWindows()
    env.close()
    return total_reward


if __name__ == "__main__":
    # 1) recompute normalization and scale
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    all_obs = np.concatenate([ep.observations[:-1] for ep in episodes], axis=0)
    state_mean = all_obs.mean(axis=0).astype(np.float32)
    state_std  = all_obs.std(axis=0).astype(np.float32) + 1e-6
    all_rets = [float(ep.rewards.sum()) for ep in episodes]
    scale = max(all_rets)
    print(f"state_mean/std and scale = {scale:.3f}")

    # 2) load model
    state_dim = episodes[0].observations.shape[1]
    act_dim   = episodes[0].actions.shape[1]
    model = DecisionTransformer(
        state_dim  = state_dim,
        act_dim    = act_dim,
        max_length = MAX_LENGTH,
        transformer_name = "gpt2",
        action_tanh = True,
        state_mean  = torch.from_numpy(state_mean),
        state_std   = torch.from_numpy(state_std),
        scale       = scale,
    ).to(device)
    ckpt = torch.load("dt_adroit_door_epoch10.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 3) choose a target_return (e.g., max expert scale)
    target_return = 0 #scale + 9000000
    out_video = "adroit_replay.mp4"
    if os.path.exists(out_video):
        os.remove(out_video)

    ret = record_episode(model, state_mean, state_std, scale, target_return,
                         video_path=out_video, fps=25)
    print(f"Episode return: {ret:.2f} — saved replay to {out_video}")

