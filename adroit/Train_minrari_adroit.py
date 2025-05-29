
import time
import numpy as np
import torch
import torch.nn as nn
from myDataLoader import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import minari

from model_cnn import DecisionTransformerRC
from config import MAX_LENGTH, BATCH_SIZE, LR, EPOCHS, device




def main():
    # load data
    dataset = minari.load_dataset("D4RL/door/expert-v2")
    episodes = list(dataset.iterate_episodes())
    # compute stats
    all_obs = np.concatenate([np.array(ep.observations[:-1],dtype=np.float32) for ep in episodes],axis=0)
    state_mean = all_obs.mean(0); state_std = all_obs.std(0)+1e-6
    all_rets = [float(ep.rewards.sum()) for ep in episodes]; scale=max(all_rets)
    # dataloader
    ds = FullBlockVectorDataset(episodes, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # model
    model = DecisionTransformerRC(
        state_dim=39, act_dim=28, max_length=MAX_LENGTH,
        transformer_name="gpt2",
        action_tanh=True,
        state_mean=torch.from_numpy(state_mean),
        state_std=torch.from_numpy(state_std),
        scale=scale
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # losses
    mse = nn.MSELoss()

    writer = SummaryWriter("runs/adroit_rc")
    step=0
    for epoch in range(1, EPOCHS+1):
        epoch_a=0; epoch_r=0; epoch_n=0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            S = batch["states"].to(device)
            A = batch["actions"].to(device)
            R = batch["returns_to_go"].to(device)
            Tm= batch["timesteps"].to(device)
            mask = batch["attention_mask"].to(device)
            # normalize
            S_norm = (S - model.state_mean) / model.state_std
            # forward
            _, A_pred, R_pred = model(S_norm, A, R, Tm)
            # action loss
            loss_a = ( ((A_pred - A)**2).mean(-1) * mask ).sum() / mask.sum()
            # return loss
            loss_r = mse(R_pred, R)
            loss = loss_a + 1.0 * loss_r
            # backward
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.25); opt.step()
            # log
            writer.add_scalar("Loss/action", loss_a.item(), step)
            writer.add_scalar("Loss/return", loss_r.item(), step)
            writer.add_scalar("Loss/total", loss.item(), step)
            step+=1
            epoch_a+=loss_a.item(); epoch_r+=loss_r.item(); epoch_n+=1
        print(f"Epoch {epoch} — a_loss {epoch_a/epoch_n:.4f}, r_loss {epoch_r/epoch_n:.4f}")
        if epoch%5==0:
            torch.save(model.state_dict(), f"dt_adroit_rc_epoch{epoch}.pt")
    writer.close()

if __name__ == "__main__":
    main()
