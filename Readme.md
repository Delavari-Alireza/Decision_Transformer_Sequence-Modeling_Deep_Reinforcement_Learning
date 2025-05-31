# Decision Transformer Implementation

Offline reinforcement learning via sequence modeling with Transformers. This repository provides my own implementation of Decision Transformers—recasting trajectories of states, actions, and returns‑to‑go as token sequences and leveraging decoder‑only architectures (e.g., GPT‑2) to generate actions conditioned on desired returns and history. Experiments cover synthetic random walks, continuous control (HalfCheetah, Hopper), robotic manipulation (AdroitHandDoor), and Atari games (Space Invaders, Breakout). Inspired by [Chen et al. 2021](https://arxiv.org/abs/2106.01345).

---
[//]: # ( ### $${\color{lightgreen}Checkout \space \space Subfolders \space For \space Better \space Explanations \space and \space visual \space examples! }$$ )
> ✅ **Checkout subfolders for better explanations and visual examples!**

---

## Overview

A Decision Transformer reframes offline reinforcement learning as a sequence modeling task: trajectories of states, actions, and returns-to-go (RTG) are treated like token sequences, and a decoder‑only Transformer predicts the next action conditioned on past observations and a target return. You can swap out the backbone by simply changing the Hugging Face model name:

```python
DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    max_length=MAX_LENGTH,
    transformer_name="gpt2",  # or any decoder model on Hugging Face
    action_tanh=True,
    state_mean=torch.from_numpy(state_mean),
    state_std=torch.from_numpy(state_std),
    scale=scale
)
```

This flexibility allows leveraging powerful pretrained decoders (GPT‑2, GPT‑Neo, etc.) for RL tasks without modifying core code.

---

## Datasets

* **Graph:** Synthetic random-walk trajectories (no external dataset) used to show that a Decision Transformer can learn return predictions purely from sequence structure.
* **HalfCheetah-v5:** Expert demonstrations from Hugging Face dataset `edbeeching/decision_transformer_gym_replay`, variant `halfcheetah-expert-v2`.
* **AdroitHandDoor-v1:** Expert trajectories via MINARI (`D4RL/door/expert-v2`).
* **Space Invaders:** Expert Atari data from MINARI (`atari/spaceinvaders/expert-v0`).
* **Breakout:** Expert Atari data from MINARI (`atari/breakout/expert-v0`).
* **Hopper-v4:** Pretrained Decision Transformer from Hugging Face (`edbeeching/decision-transformer-gym-hopper-expert`).

## Repository Structure

* **`graph/`**
  Illustrates that a Decision Transformer can extract meaningful return predictions even from random-walk data. Random trajectories are fed in, and the model learns to forecast returns‑to‑go—highlighting the power of sequence modeling on synthetic data. See `graph/README.md` for plots and videos.

* **`HalfCheetah/`**
  Offline sequence modeling on HalfCheetah-v5 using expert trajectories from Hugging Face. Includes training scripts, evaluation rollouts, RTG‑vs‑reward comparison plots, and videos demonstrating how varying RTG influences performance. See `HalfCheetah/README.md`.

* **`adroit/`**
  CNN‑based Decision Transformer applied to the AdroitHandDoor-v1 task via MINARI. Covers data loading, model architectures (with and without CNN), RTG sweeps, and expert vs. learned behavior videos. Highlights dependency on dataset diversity—expert-only data yields similar returns across RTGs. See `adroit/README.md`.

* **`spaceVader/`**
  Image‑based Decision Transformer on Atari Space Invaders. Uses a CNN encoder for frames, with RTG-conditioning experiments and performance plots. Demonstrates the transformer’s capacity on visual tasks. See `spaceVader/README.md`.

* **`breakout/`**
  Breakout experiments showcasing the impact of limited expert data—model often fails to learn ball control, returning zero reward regardless of RTG. Underlines need for larger, diverse datasets. See `breakout/README.md`.

* **`hopper_huggingFace/`**
  Reference Hugging Face Decision Transformer on Hopper-v4. Provides RTG vs. reward sweeps and rollout videos for direct comparison, illustrating that best performance sometimes arises from sub‑maximal RTGs. See `hopper_huggingFace/README.md`.

---

## Requirements

* **Python:** 3.11
* **Dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

---

## Hardware

Experiments ran on **NVIDIA GeForce GTX 1050 (4 GB)**. High-dimensional visual tasks (Atari) occasionally required CPU due to GPU memory limits, but GPU mode is supported when resources permit.

---

## Getting Started

1. **Clone** the repository and set up a Python 3.11 environment.
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. **Explore** any subfolder (e.g., `HalfCheetah/`) and follow its own `README.md` for detailed instructions on training, evaluation, and visualization.

---

Enjoy seeing how Transformers, with just return‑conditioning and autoregressive decoding, can tackle offline RL by treating it as sequence modeling!
