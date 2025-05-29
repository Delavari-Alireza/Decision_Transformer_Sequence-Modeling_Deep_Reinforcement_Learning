
import numpy as np
import gymnasium as gym
from datasets import load_dataset

# 1) Load the trajectory
dataset = load_dataset("edbeeching/decision_transformer_gym_replay",
                       "halfcheetah-expert-v2")
traj   = dataset["train"][0]
obs    = np.array(traj["observations"])   # (T, obs_dim=17)
acts   = np.array(traj["actions"])        # (T, act_dim=6)
rews   = np.array(traj["rewards"])        # (T,)
dones  = np.array(traj["dones"])          # (T,)
offset = 1 if len(obs) == len(acts) + 1 else 0
T      = len(acts)

print(f"""

trajectory length: {T}
trajectory offset: {offset}
obs.shape: {obs.shape}
acts.shape: {acts.shape}
rews.shape: {rews.shape}
dones.shape: {dones.shape}

""")


class VisualTrajectoryEnv(gym.Env):
    def __init__(self, obs, acts, rews, dones, offset):
        # -- underlying env used for rendering & done flags
        self.base = gym.make("HalfCheetah-v5", render_mode="human")
        self.observation_space = self.base.observation_space
        self.action_space      = self.base.action_space

        # splitting obs into joint-angles vs velocities
        self.n_qpos  = self.base.unwrapped.model.nq
        self.pos_dim = self.n_qpos - 1              # dataset obs excludes root-x
        # initial root-x from env defaults
        self.root_x0 = float(self.base.unwrapped.init_qpos[0])
        # timestep for integration
        self.dt      = float(self.base.unwrapped.model.opt.timestep)

        self.obs    = obs
        self.acts   = acts
        self.rews   = rews
        self.dones  = dones
        self.offset = offset
        self.t      = 0
        self.root_x = None

    def reset(self):
        # 1) reset the base so rendering is enabled
        _, _ = self.base.reset()

        # 2) initialize root_x and simulator state to first obs
        self.root_x = self.root_x0
        pos_obs = self.obs[0, :self.pos_dim]
        vel_obs = self.obs[0, self.pos_dim:]
        qpos    = np.concatenate(([self.root_x], pos_obs))
        qvel    = vel_obs
        self.base.unwrapped.set_state(qpos, qvel)

        # 3) render the exact first frame
        self.base.render()

        self.t = 0
        return self.obs[0], {}

    def step(self, action):
        # sanity-check: using the expert action
        assert np.allclose(action, self.acts[self.t]), f"action mismatch @ t={self.t}"

        # 1) figure out which obs to display next
        idx     = self.t + self.offset
        pos_obs = self.obs[idx, :self.pos_dim]
        vel_obs = self.obs[idx, self.pos_dim:]

        # 2) integrate root-x using the recorded root velocity
        root_vel = vel_obs[0]
        self.root_x = self.root_x + root_vel * self.dt

        # 3) build full qpos/qvel and override simulator
        qpos = np.concatenate(([self.root_x], pos_obs))
        qvel = vel_obs
        self.base.unwrapped.set_state(qpos, qvel)

        # 4) render and step internally to get done/truncated
        self.base.render()
        _, _, done_env, truncated, _ = self.base.step(action)

        # 5) return the dataset’s reward & done flag
        r    = float(self.rews[self.t])
        done = bool(self.dones[self.t]) or done_env
        self.t += 1

        return self.obs[idx], r, done, truncated, {}

    def close(self):
        self.base.close()


# 3) Instantiate and play back
env = VisualTrajectoryEnv(obs, acts, rews, dones, offset)
o, _ = env.reset()
for t in range(T):
    o, r, done, truncated, _ = env.step(acts[t])
    if done or truncated:
        break
env.close()
