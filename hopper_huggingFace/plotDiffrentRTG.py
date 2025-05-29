import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])]).to(dtype=torch.long, device=device).reshape(1, -1)

    states = torch.cat([torch.zeros((1, padding, model.config.state_dim), device=device), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim), device=device), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=device), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long, device=device), timesteps], dim=1)

    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
    return action_preds[0, -1]


def evaluate_episode(model, env, state_mean, state_std, rtg, scale=1000.0, max_ep_len=1000):
    obs, _ = env.reset()
    episode_return = 0
    state = torch.from_numpy(obs).reshape(1, -1).to(device).float()
    states = state
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(rtg / scale, device=device).reshape(1, 1)
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

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = torch.from_numpy(obs).to(device).reshape(1, -1).float()
        states = torch.cat([states, state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        if done:
            break

    return episode_return


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Hopper-v4")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load pretrained DT
model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-expert")
model = model.to(device)

# mean/std from model card
state_mean = np.array([1.3490015, -0.11208222, -0.5506444, -0.13188992, -0.00378754,
                       2.6071432, 0.02322114, -0.01626922, -0.06840388, -0.05183131, 0.04272673])
state_std = np.array([0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333,
                      0.5899924, 1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027])
state_mean = torch.tensor(state_mean).to(device)
state_std = torch.tensor(state_std).to(device)

# Run for different RTG targets
rtg_values = list(range(0, 4001, 400))  # 0 to 4000
episode_returns = []

for rtg in rtg_values:
    print(f"Evaluating RTG = {rtg}")
    ep_ret = evaluate_episode(model, env, state_mean, state_std, rtg)
    print(f"→ Return: {ep_ret:.2f}")
    episode_returns.append(ep_ret)

env.close()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(rtg_values, episode_returns, marker='o')
plt.xlabel("Target Return (RTG)")
plt.ylabel("Achieved Return")
plt.title("Decision Transformer: Return vs RTG (Hopper-v4)")
plt.grid(True)
plt.savefig("dt_rtg_vs_reward_hopper.png")
print("Plot saved as dt_rtg_vs_reward_hopper.png")
