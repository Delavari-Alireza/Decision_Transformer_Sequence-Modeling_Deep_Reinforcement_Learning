import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

class GraphEnv(gym.Env):
    metadata = {"render_fps": 2}

    def __init__(self, G: nx.Graph, goal: int, horizon: int = 50, start_state=None):
        super().__init__()
        self.G       = G.copy()
        self.goal    = int(goal)
        self.horizon = int(horizon)
        self.n_nodes = G.number_of_nodes()


        self.start_state = start_state
        if self.start_state is not None:
            assert self.start_state in self.G.nodes, \
                f"start_state {self.start_state} is not a valid node in the graph."

        self.observation_space = spaces.Discrete(self.n_nodes)
        self.action_space      = spaces.Discrete(self.n_nodes)

        # fixed layout
        self.pos = nx.spring_layout(self.G, seed=42)

        # Matplotlib figure + canvas
        self.fig    = plt.Figure(figsize=(4,4))
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax     = self.fig.add_subplot(111)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.start_state if self.start_state!=None else int(self.np_random.integers(0, self.n_nodes))
        self.steps = 0
        return self.current_node, {}

    def step(self, action):
        if self.G.has_edge(self.current_node, action):
            self.current_node = int(action)
        if self.current_node == self.goal:
            reward = 0.0
        else:
            reward = -1.0
        self.steps += 1
        done = (self.current_node == self.goal) or (self.steps >= self.horizon)
        return self.current_node, reward, done, False, {}


    def render(self, return_rgb=False):
        # draw
        self.ax.clear()
        colors = [
            "orange" if n == self.current_node
            else "magenta" if n == self.goal
            else "skyblue"
            for n in self.G.nodes()
        ]
        nx.draw(self.G, pos=self.pos, ax=self.ax, with_labels=True,
                node_color=colors, node_size=300, font_size=10)

        self.canvas.draw()
        buf = self.canvas.tostring_argb()
        w, h = self.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgba = arr[:, :, [1, 2, 3, 0]]
        rgb = rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if not return_rgb:
            cv2.imshow("GraphEnv", bgr)
            if cv2.waitKey(int(1000 / self.metadata["render_fps"])) & 0xFF == ord('q'):
                cv2.destroyWindow("GraphEnv")

        return bgr if return_rgb else None

    def close(self):
        cv2.destroyAllWindows()
        plt.close(self.fig)

    def valid_actions(self):
        """
        Returns the list of neighbor node‐ids you can actually move to
        from the current node.
        """
        return list(self.G.neighbors(self.current_node))



import random
import pickle
import networkx as nx

def build_graph():
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(i, i+1) for i in range(9)])
    G.add_edges_from([(0,5), (2,7), (3,8), (1,4)])
    return G

def generate_dataset(num_episodes=10000, horizon=20, seed=0):
    random.seed(seed)

    G   = build_graph()
    goal = 9
    env = GraphEnv(G, goal=goal, horizon=horizon)

    episodes = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        while obs == goal:
            obs, _ = env.reset()      # s₀
        traj = {"states": [], "actions": [], "rewards": []}

        for t in range(horizon):
            valid = env.valid_actions()
            action = random.choice(valid)      # aₜ
            next_obs, reward, done, _, _ = env.step(action)  # rₜ, sₜ₊₁
            # store time-step t
            traj["states"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(reward)

            obs = next_obs
            if done:
                break

        episodes.append(traj)

    env.close()
    return episodes

if __name__ == "__main__":
    data = generate_dataset(num_episodes=10000, horizon=50, seed=42)
    # save to disk
    with open("graph_random_10000.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} episodes to graph_random_100.pkl")




