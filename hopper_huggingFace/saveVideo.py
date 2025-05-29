import torch
import gymnasium as gym
import numpy as np
import cv2
from transformers import DecisionTransformerModel


def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]

    device = states.device
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])]) \
                        .to(dtype=torch.long, device=device).reshape(1, -1)

    states = torch.cat([torch.zeros((1, padding, model.config.state_dim), device=device), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim), device=device), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=device), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long, device=device), timesteps], dim=1)

    _, action_preds, _ = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


# === ENV SETUP ===
env = gym.make("Hopper-v4", render_mode="rgb_array")  # << updated
obs, _ = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_ep_len = 1000
scale = 1000.0
TARGET_RETURN = 10 / scale

# mean and std from model card
state_mean = np.array([1.3490015, -0.11208222, -0.5506444, -0.13188992, -0.00378754,
                       2.6071432, 0.02322114, -0.01626922, -0.06840388, -0.05183131, 0.04272673])
state_std = np.array([0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333,
                      0.5899924, 1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027])
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert")
model = model.to(device)
state_mean = torch.from_numpy(state_mean).to(device)
state_std = torch.from_numpy(state_std).to(device)

# === VIDEO SETUP ===
video_path = "hopper_dt.mp4"
video_writer = None
fps = 30

# === ROLLOUT ===
episode_return, episode_length = 0, 0
state, _ = env.reset()
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device, dtype=torch.float32)
actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(0, device=device, dtype=torch.float32)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

for t in range(max_ep_len):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    actions[-1] = action
    action = action.detach().cpu().numpy()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # === GET FRAME & WRITE TO VIDEO ===
    frame = env.render()
    if video_writer is None:
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cur_state = torch.from_numpy(state).to(device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1

    if done:
        break

env.close()
if video_writer:
    video_writer.release()

print(f"Episode return: {episode_return:.2f}")
print(f"Saved video to: {video_path}")
