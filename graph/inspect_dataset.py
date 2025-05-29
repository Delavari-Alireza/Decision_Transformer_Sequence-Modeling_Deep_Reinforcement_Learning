import pickle
import pprint
import numpy as np

# 1) load the file
with open("graph_random_10000.pkl", "rb") as f:
    episodes = pickle.load(f)


# 2) top-level summary
print(f"Total episodes: {len(episodes)}")
print("Keys in each episode:", list(episodes[0].keys()))

# 3) per-episode lengths (should all be ≤ horizon)
lengths = [len(ep["states"]) for ep in episodes]
print(f"Min length: {min(lengths)}, max length: {max(lengths)}")

# 4) peek at the first episode
print("\nEpisode 0 preview:")

pprint.pprint({
    "states ":  episodes[0]["states"],
    "actions ": episodes[0]["actions"],
    "rewards ": episodes[0]["rewards"],
    "total reward":       sum(episodes[0]["rewards"]),
    "returns_to_go": np.cumsum(episodes[0]["rewards"][::-1])[::-1].tolist(),
})

# 5) (optional) full distribution of rewards
total_rewards = [sum(ep["rewards"]) for ep in episodes]
print(f"\nTotal‐reward stats: min={min(total_rewards):.1f}, max={max(total_rewards):.1f}, avg={sum(total_rewards)/len(total_rewards):.1f}")
