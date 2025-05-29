import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import minari
from config import MAX_LENGTH, BATCH_SIZE, LR, EPOCHS, device
from model_cnn import  DecisionTransformerRC , DecisionTransformerCNN
from myDataLoader import *
import gc

device = 'cpu'   # avoid out of memory
BATCH_SIZE = 2
MAX_LENGTH = 15
def main():
    # load dataset & compute RTG scale
    dataset = minari.load_dataset("atari/breakout/expert-v0")
    # episodes = list(dataset.iterate_episodes())
    # scale = max(float(ep.rewards.sum()) for ep in episodes)
    # compute stats
    # all_obs = np.concatenate([np.array(ep.observations[:-1],dtype=np.float32) for ep in episodes],axis=0)
    # state_mean = all_obs.mean(0); state_std = all_obs.std(0)+1e-6
    # Lazily compute scale from rewards
    scale = 0.0
    for ep in dataset.iterate_episodes():
        if len(ep.actions) >= MAX_LENGTH:
            ep_return = float(ep.rewards.sum())
            if ep_return > scale:
                scale = ep_return

    # dims
    spec = dataset.spec
    C,H,W = spec.observation_space.shape[2], spec.observation_space.shape[0], spec.observation_space.shape[1]
    act_dim = spec.action_space.n

    # DataLoader
    ds = FullBlockDataset(dataset, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, drop_last=False)

    # model + optimizer + CE loss
    model = DecisionTransformerCNN(
        state_shape=(C,H,W), act_dim=act_dim,
        max_length=MAX_LENGTH, transformer_name="gpt2",
        scale=scale
    ).to(device)

    # model = DecisionTransformerRC(
    #     state_dim=39, act_dim=28, max_length=MAX_LENGTH,
    #     transformer_name="gpt2",
    #     action_tanh=True,
    #     state_mean=torch.from_numpy(state_mean),
    #     state_std=torch.from_numpy(state_std),
    #     scale=scale
    # ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    writer = SummaryWriter("runs/minari_bearkout")
    step = 0
    model.train()

    for epoch in range(1, EPOCHS+1):
        gc.collect()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            S = batch['states'].to(device)          # (B,T,3,H,W)
            A = batch['actions'].to(device)         # (B,T)
            R = batch['returns_to_go'].to(device)   # (B,T,1)
            Tm= batch['timesteps'].to(device)       # (B,T)
            B = S.shape[0]

            mask = torch.ones(B, MAX_LENGTH, dtype=torch.bool, device=device)

            logits, _ = model(S, A, R, Tm, mask)
            # logits: (B,T,act_dim)

            loss = ce_loss(
                logits.view(-1, act_dim),
                A.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            writer.add_scalar("Loss/train_batch", loss.item(), step)
            step += 1
            epoch_loss += loss.item()

        avg = epoch_loss/len(loader)
        print(f"Epoch {epoch}/{EPOCHS} — loss {avg:.4f} — time {time.time()-t0:.1f}s")
        writer.add_scalar("Loss/train_epoch", avg, epoch)
        if epoch%5==0:
            torch.save(model.state_dict(), f"dt_cnn_no_pad_epoch_cpu_dis{epoch}.pt")
    writer.close()

if __name__=="__main__":
    main()
