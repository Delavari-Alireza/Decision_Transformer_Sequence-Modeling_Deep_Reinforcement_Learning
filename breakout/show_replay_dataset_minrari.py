import numpy as np
import minari
import cv2

# 1) load the expert SpaceInvaders dataset
dataset = minari.load_dataset("atari/breakout/expert-v0")

# 2) grab the first full episode
episode = next(dataset.iterate_episodes())
obs    = episode.observations    # shape (T+1, H, W, 3), uint8 RGB
acts   = episode.actions         # shape (T,)
rews   = episode.rewards         # shape (T,)
dones  = episode.terminations    # shape (T,)
T      = acts.shape[0]

print(f"T={T}, obs.shape={obs.shape}, acts.shape={acts.shape}")

# 3) replay with OpenCV
win = "SpaceInvaders Expert Replay"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
for t in range(T):
    frame = obs[t]
    # convert RGB → BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # overlay action/reward text
    text = f"t={t:4d}  a={acts[t]:2d}  r={rews[t]:.2f}"
    cv2.putText(frame_bgr, text, (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,
                lineType=cv2.LINE_AA)
    # show
    cv2.imshow(win, frame_bgr)
    key = cv2.waitKey(40)  # ~25 FPS
    if key == ord('q') or dones[t]:
        print(f"stopping at t={t}, done={dones[t]}")
        break

cv2.destroyAllWindows()
