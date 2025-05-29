
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import (
    MAX_LENGTH, BATCH_SIZE, LR, EPOCHS, device,
    state_mean, state_std, scale
)
from DataLoader import TrajectoryDataset, collate_fn
from model import DecisionTransformer

def main():
    # 1) load trajectories & dims
    from datasets import load_dataset
    hf = load_dataset("edbeeching/decision_transformer_gym_replay",
                      "halfcheetah-expert-v2")["train"]
    sample = hf[0]
    state_dim = np.array(sample["observations"], dtype=np.float32).shape[1]
    act_dim   = np.array(sample["actions"],      dtype=np.float32).shape[1]
    print(f"State dim: {state_dim}, Action dim: {act_dim}")

    # 2) DataLoader
    ds     = TrajectoryDataset(hf, max_length=MAX_LENGTH)
    loader = DataLoader(ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=collate_fn,
                        drop_last=True)

    # 3) Model + optimizer + loss
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=MAX_LENGTH,
        transformer_name="gpt2",
        action_tanh=True,
        state_mean=torch.from_numpy(state_mean),
        state_std =torch.from_numpy(state_std),
        scale=scale
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse_loss  = nn.MSELoss(reduction="none")

    # 4) TensorBoard writer
    writer = SummaryWriter(log_dir="runs/decision_transformer_expt")
    global_step = 0

    model.train()
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            states        = batch["states"].to(device)        # (B, T, S)
            actions       = batch["actions"].to(device)       # (B, T, A)
            returns_to_go = batch["returns_to_go"].to(device) # (B, T, 1)
            timesteps     = batch["timesteps"].to(device)     # (B, T)
            mask          = batch["attention_mask"].to(device) # (B, T)

            # forward
            _, action_preds, _ = model(
                states, actions, returns_to_go, timesteps
            )

            # compute masked MSE
            B, T, A = action_preds.shape
            loss_per_step = mse_loss(action_preds, actions).mean(-1)  # (B, T)
            loss = (loss_per_step * mask).sum() / mask.sum()

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
            optimizer.step()

            # log per‐batch
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            writer.add_scalar("LearningRate",
                              optimizer.param_groups[0]['lr'],
                              global_step)
            global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix(train_loss=epoch_loss/(pbar.n+1))

        # end of epoch
        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - start_time
        print(f"Epoch {epoch:2d}/{EPOCHS:2d} — loss: {avg_loss:.6f} — time: {elapsed:.1f}s")

        # log per‐epoch
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        # log histogram of action-prediction error
        with torch.no_grad():
            err = (action_preds - actions).detach().cpu().view(-1)
            writer.add_histogram("ActionError", err, epoch)

        # checkpoint
        if epoch % 5 == 0:
            ckpt = f"dt_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f" → saved checkpoint: {ckpt}")

        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()
