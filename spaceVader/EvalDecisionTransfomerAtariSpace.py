import numpy as np
import torch
import gymnasium as gym
import ale_py
# import gymnasium.envs.atari
import cv2
import minari

from model_cnn import DecisionTransformerCNN
from config    import MAX_LENGTH, device

def main():
    # 1) Load Minari to compute the same RTG scale used in training
    video_out_path = "dt_agent_spaceinvaders.mp4"
    fps = 30
    video_writer = None

    dataset = minari.load_dataset("atari/spaceinvaders/expert-v0")
    episodes = list(dataset.iterate_episodes())
    scale    = max(float(ep.rewards.sum()) for ep in episodes)
    print(f"Using RTG scale = {scale:.1f}")

    # 2) Create the Atari env in rgb_array mode
    env = gym.make(
        "SpaceInvaders-v4",
        render_mode="rgb_array",
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0,
        full_action_space=False,
        max_num_frames_per_episode=108000,
    )
    obs, _ = env.reset()  # (210,160,3) uint8

    # 3) Instantiate trained model
    C, H, W   = obs.shape[2], obs.shape[0], obs.shape[1]
    act_dim   = env.action_space.n
    model = DecisionTransformerCNN(
        state_shape      = (C,H,W),
        act_dim          = act_dim,
        max_length       = MAX_LENGTH,
        transformer_name = "gpt2",
        state_mean       = None,
        state_std        = None,
        scale            = scale,         # <— Important!
    ).to(device)

    ckpt = torch.load("dt_cnn_no_pad_epoch15.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 4) Prepare rolling buffers of length MAX_LENGTH
    first_frame = (obs.astype(np.float32)/255.0).transpose(2,0,1)  # (C,H,W)
    state_hist  = [first_frame.copy() for _ in range(MAX_LENGTH)]
    # action_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(MAX_LENGTH)]
    action_hist = [0 for _ in range(MAX_LENGTH)]
    ts_hist     = list(range(MAX_LENGTH))  # 0,1,2…T-1

    # 4a) Seed RTG with your *raw* target return (game points)
    target_return = 10
    returns_hist  = [float(target_return)] * MAX_LENGTH

    total_reward = 0.0
    done = False

    cv2.namedWindow("DT-Agent", cv2.WINDOW_NORMAL)
    while not done:
        # 5) Build a batch of size 1
        S   = torch.from_numpy(np.stack(state_hist, 0)) \
                    .unsqueeze(0).to(device)           # (1, T, C, H, W)
        A   = torch.from_numpy(np.stack(action_hist, 0)) \
                    .unsqueeze(0).to(device)           # (1, T, act_dim)
        R   = torch.tensor(returns_hist, dtype=torch.float32) \
                    .unsqueeze(0).unsqueeze(-1).to(device)  # (1, T, 1)
        Tm  = torch.tensor(ts_hist, dtype=torch.long) \
                    .unsqueeze(0).to(device)           # (1, T)
        mask= torch.ones(1, MAX_LENGTH, dtype=torch.bool, device=device)  # no padding

        # 6) Forward → get action logits
        with torch.no_grad():
            _, logits, _ = model(S, A, R, Tm, mask)

        # 7) Pick the highest‐prob action
        a = int(logits[0, -1].argmax().cpu().numpy())

        # 8) Step the env
        obs, rew, term, trunc, _ = env.step(a)
        total_reward += float(rew)
        done = term or trunc

        # 9) Render via OpenCV
        # frame_vis = env.render()  # (210,160,3) uint8
        # cv2.imshow("DT-Agent", frame_vis)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_vis = env.render()  # (210,160,3) uint8

        # Initialize video writer
        if video_writer is None:
            height, width, _ = frame_vis.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' for .avi
            video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

        video_writer.write(frame_vis)  # Save the frame
        cv2.imshow("DT-Agent", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 10) Update rolling buffers
        #   a) state
        f = (obs.astype(np.float32)/255.0).transpose(2,0,1)
        state_hist  = state_hist[1:]  + [f]
        #   b) action
        oh = np.zeros(act_dim, dtype=np.float32)
        oh[a] = 1.0
        # action_hist = action_hist[1:] + [oh]
        action_hist = action_hist[1:] + [a]
        #   c) return-to-go
        new_rtg     = returns_hist[-1] - float(rew)
        returns_hist= returns_hist[1:] + [new_rtg]
        #   d) timestep
        ts_next     = (ts_hist[-1] + 1) % MAX_LENGTH
        ts_hist     = ts_hist[1:]      + [ts_next]
    if video_writer:
        video_writer.release()

    print(f"\n*** Episode return: {total_reward:.2f} ***")
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
