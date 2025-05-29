import numpy as np
import minari
import cv2

# 1) load the expert AdroitHandDoor dataset
dataset = minari.load_dataset("D4RL/door/expert-v2")
episode = next(dataset.iterate_episodes())
obs    = episode.observations    # shape (T+1, 39), float64
acts   = episode.actions         # shape (T, 28), float32
rews   = episode.rewards         # shape (T,), float32
dones  = episode.terminations    # shape (T,), bool
T      = acts.shape[0]

print(f"Episode length: T={T}")
print(f"obs.shape={obs.shape}, acts.shape={acts.shape}")

# 2) recover the environment used to collect the data
try:
    env = dataset.recover_environment(render_mode="rgb_array")
except TypeError:
    env = dataset.recover_environment()
obs_env, _ = env.reset()

# 3) Setup video writer
frame0 = env.render()  # get one frame to get size
height, width, _ = frame0.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("adroit_expert_replay.mp4", fourcc, 25, (width, height))

win = "AdroitHandDoor Expert Replay"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

for t in range(T):
    action = acts[t]
    obs_env, reward_env, done_env, truncated, info = env.step(action)
    frame = env.render()  # RGB array
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    text = f"t={t:4d}  r={rews[t]:.3f}  done={done_env}"
    cv2.putText(frame_bgr, text, (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,
                lineType=cv2.LINE_AA)
    video_writer.write(frame_bgr)   # Write frame to video file
    cv2.imshow(win, frame_bgr)

    key = cv2.waitKey(40)  # ~25 FPS
    if key == ord('q') or done_env:
        print(f"Stopping at t={t}, done_env={done_env}")
        break

video_writer.release()
cv2.destroyAllWindows()
env.close()
print("Video saved as adroit_expert_replay.mp4")
