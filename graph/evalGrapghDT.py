import cv2
import torch
import networkx as nx
from graph_env import GraphEnv
from model_graph import DecisionTransformerRCGraph
from config import MAX_LENGTH, device

def eval_graph_with_masking(start_state=None, RTG=0 , render_env=True):


    # 1. Construct the graph
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(i, i + 1) for i in range(9)])
    G.add_edges_from([(0, 5), (2, 7), (3, 8), (1, 4)])
    goal = 9

    # 2. Initialize the environment
    env = GraphEnv(G, goal=goal, horizon=100 , start_state=start_state)
    obs, _ = env.reset()

    # 3. Load the trained model
    model = DecisionTransformerRCGraph(
        G.number_of_nodes(),
        MAX_LENGTH,
        transformer_name="gpt2"
    ).to(device)
    # ckpt = torch.load("dt_graph_masked.pt", map_location=device)
    ckpt = torch.load("dt_graph_masked_rtg4.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 4. Initialize history
    # state_hist   = [obs] * MAX_LENGTH
    # action_hist  = [0]   * MAX_LENGTH
    # returns_hist = [0.0] * MAX_LENGTH
    # ts_hist      = list(range(MAX_LENGTH))
    # total_r      = 0.0
    actual_start = obs if start_state is None else start_state

    state_hist = [0] * (MAX_LENGTH - 1) + [actual_start]
    action_hist = [0] * MAX_LENGTH
    # returns_hist = [0] * (MAX_LENGTH - 1) + [RTG]
    returns_hist = [RTG - i for i in range(MAX_LENGTH - 1, 0, -1)] + [RTG]

    ts_hist = list(range(MAX_LENGTH))
    total_r = 0.0
    # 5. Rollout
    for t in range(env.horizon):
        valid = env.valid_actions()

        # Build input tensors
        S  = torch.tensor(state_hist,   dtype=torch.long,   device=device).unsqueeze(0)
        A  = torch.tensor(action_hist,  dtype=torch.long,   device=device).unsqueeze(0)
        R  = torch.tensor(returns_hist, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        Tm = torch.tensor(ts_hist,      dtype=torch.long,   device=device).unsqueeze(0)

        # Construct valid_mask
        vm = torch.zeros(1, MAX_LENGTH, G.number_of_nodes(), dtype=torch.bool, device=device)
        for a in valid:
            if a != state_hist[-1]:  # disallow staying at the same node
                vm[0, -1, a] = True

        # Run model with masking
        with torch.no_grad():
            logits , _ = model(S, A, R, Tm, vm)  # shape (1, T, n_nodes)

        pred  = int(logits[0, -1].argmax().item())

        # Fallback if model still outputs invalid action (e.g. due to numerical issues)
        if pred not in valid or pred == state_hist[-1]:
            pred = env.np_random.choice([a for a in valid if a != state_hist[-1]])

        action = pred

        # Take the action in the environment
        obs, reward, done, trunc, _ = env.step(action)
        total_r += reward
        if render_env:
            env.render()

        print(f"Step {t:2d} | Node {obs} | Act {action} | R {reward:.1f} | RTG {returns_hist[-1]:.1f}")

        if done:
            break

        # Update history buffers
        state_hist   = state_hist[1:]   + [obs]
        action_hist  = action_hist[1:]  + [action]
        # returns_hist = returns_hist[1:] + [returns_hist[-1] + reward]
        new_rtg = returns_hist[-1] - reward  # subtract reward to move toward 0
        if new_rtg>=0:
            new_rtg = 0
        returns_hist = returns_hist[1:] + [new_rtg]
        print(f"→ Updated RTG: {new_rtg:.1f}")
        print(f"→ Updated RTG: {returns_hist}")
        ts_hist      = ts_hist[1:]      + [(ts_hist[-1] + 1) % MAX_LENGTH]

    print(f"\nEpisode ended at step {t}, total reward = {total_r:.1f}")
    env.close()




def eval_graph_with_masking_video(start_state=None, RTG=0, render_env=True, save_video=False, video_path="graph_episode.mp4"):
    import torch
    import networkx as nx
    from graph_env import GraphEnv
    from model_graph import DecisionTransformerRCGraph
    from config import MAX_LENGTH, device

    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(i, i + 1) for i in range(9)])
    G.add_edges_from([(0, 5), (2, 7), (3, 8), (1, 4)])
    goal = 9

    env = GraphEnv(G, goal=goal, horizon=100, start_state=start_state)
    obs, _ = env.reset()

    model = DecisionTransformerRCGraph(
        G.number_of_nodes(),
        MAX_LENGTH,
        transformer_name="gpt2"
    ).to(device)

    ckpt = torch.load("dt_graph_masked_rtg4.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    actual_start = obs if start_state is None else start_state
    state_hist = [0] * (MAX_LENGTH - 1) + [actual_start]
    action_hist = [0] * MAX_LENGTH
    # returns_hist = [0] * (MAX_LENGTH - 1) + [RTG]
    returns_hist = [RTG - i for i in range(MAX_LENGTH - 1, 0, -1)] + [RTG]
    ts_hist = list(range(MAX_LENGTH))
    total_r = 0.0

    frames = []
    writer = None

    for t in range(env.horizon):
        valid = env.valid_actions()

        S = torch.tensor(state_hist, dtype=torch.long, device=device).unsqueeze(0)
        A = torch.tensor(action_hist, dtype=torch.long, device=device).unsqueeze(0)
        R = torch.tensor(returns_hist, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        Tm = torch.tensor(ts_hist, dtype=torch.long, device=device).unsqueeze(0)

        vm = torch.zeros(1, MAX_LENGTH, G.number_of_nodes(), dtype=torch.bool, device=device)
        for a in valid:
            if a != state_hist[-1]:
                vm[0, -1, a] = True

        with torch.no_grad():
            logits , _ = model(S, A, R, Tm, vm)

        pred = int(logits[0, -1].argmax().item())
        if pred not in valid or pred == state_hist[-1]:
            pred = env.np_random.choice([a for a in valid if a != state_hist[-1]])

        action = pred
        obs, reward, done, trunc, _ = env.step(action)
        total_r += reward

        if render_env or save_video:
            frame = env.render(return_rgb=True)

            if save_video:
                if writer is None:
                    height, width, _ = frame.shape
                    writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        env.metadata["render_fps"],
                        (width, height)
                    )
                writer.write(frame)

        print(f"Step {t:2d} | Node {obs} | Act {action} | R {reward:.1f} | RTG {returns_hist[-1]:.1f}")

        if done:
            break

        state_hist = state_hist[1:] + [obs]
        action_hist = action_hist[1:] + [action]
        new_rtg = returns_hist[-1] - reward
        if new_rtg >= 0:
            new_rtg = 0
        returns_hist = returns_hist[1:] + [new_rtg]
        ts_hist = ts_hist[1:] + [(ts_hist[-1] + 1) % MAX_LENGTH]

    print(f"\nEpisode ended at step {t}, total reward = {total_r:.1f}")

    if writer:
        writer.release()
        print(f"Video saved to {video_path}")

    env.close()



import matplotlib.pyplot as plt

def run_rtg_sweep(start_state=None, rtg_range=range(0, -51, -5)):
    rewards = []

    for rtg in rtg_range:
        print(f"\n>>> Evaluating with RTG = {rtg}, start_state = {start_state}")
        import io
        import sys

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        eval_graph_with_masking(start_state=start_state, RTG=rtg, render_env=False)

        sys.stdout = old_stdout
        output = mystdout.getvalue()
        for line in reversed(output.splitlines()):
            if "total reward" in line:
                total_r = float(line.split('=')[-1])
                rewards.append(total_r)
                break

    return list(rtg_range), rewards


def plot_rtg_vs_reward():
    rtg_range = list(range(0, -51, -5))

    # 1. Fixed start state
    x1, y1 = run_rtg_sweep(start_state=2)

    # 2. Random start state
    x2, y2 = run_rtg_sweep(start_state=None)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, marker='o', label='Start = 2 (Fixed)')
    plt.plot(x2, y2, marker='s', label='Start = Random')
    plt.xlabel("Input RTG")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Input RTG")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2_rtg_vs_reward_comparison.png")
    print("Plot saved as '2_rtg_vs_reward_comparison.png'")
    plt.close()




if __name__ == "__main__":
    # eval_graph_with_masking_video(start_state=1, RTG=-15 , save_video=True , video_path='state-2-RTG-15.mp4'     )
    # eval_graph_with_masking_video(start_state=1, RTG=0, save_video=True, video_path='state-2-RTG-0.mp4')
    # eval_graph_with_masking(start_state=6, RTG=0)
    plot_rtg_vs_reward()
