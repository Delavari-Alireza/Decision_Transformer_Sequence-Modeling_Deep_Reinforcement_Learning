# import pickle
# import time
# import networkx as nx
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from transformers import GPT2Model
# from tqdm import tqdm
# from myDataLoader import *
# from model_graph import DecisionTransformerRCGraph
# from  config import  *
# # -----------------------------------------------------------------------------
# # 1) hyperparams / config
# # -----------------------------------------------------------------------------
# MAX_LENGTH = 20
# BATCH_SIZE = 32
# LR         = 1e-4
# EPOCHS     = 20
# DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_FILE  = "graph_random_10000.pkl"
#
# # -----------------------------------------------------------------------------
# #  training loop
# # -----------------------------------------------------------------------------
# def main():
#     G = nx.Graph()
#     G.add_nodes_from(range(10))
#     G.add_edges_from([(i, i+1) for i in range(9)])
#     G.add_edges_from([(0,5), (2,7), (3,8), (1,4)])
#
#     # load dataset
#     with open(DATA_FILE, "rb") as f:
#         episodes = pickle.load(f)
#
#     ds     = GraphTrajectoryDataset(episodes, G, MAX_LENGTH)
#     loader = DataLoader(ds,
#                         batch_size=BATCH_SIZE,
#                         shuffle=True,
#                         collate_fn=collate_fn,
#                         drop_last=True)
#
#     model = DecisionTransformerRCGraph(
#         num_nodes   = G.number_of_nodes(),
#         max_length  = MAX_LENGTH,
#         scale       = 1.0
#     ).to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#     ce        = nn.CrossEntropyLoss(reduction="none")
#
#     writer = SummaryWriter("runs/graph_dt_masked")
#     global_step = 0
#
#     for epoch in range(1, EPOCHS+1):
#         t0, epoch_loss = time.time(), 0.0
#         model.train()
#
#         pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
#         for batch in pbar:
#             S  = batch["states"].to(DEVICE)         # (B,T)
#             A  = batch["actions"].to(DEVICE)        # (B,T)
#             R  = batch["returns_to_go"].to(DEVICE)  # (B,T,1)
#             Tm = batch["timesteps"].to(DEVICE)      # (B,T)
#             VM = batch["valid_mask"].to(DEVICE)     # (B,T,act_dim)
#             M  = batch["attention_mask"].to(DEVICE) # (B,T)
#
#             logits = model(S, A, R, Tm, VM)         # (B,T,act_dim)
#             B,T,ND = logits.shape
#
#             # compute CE only where mask==1
#             logits_flat = logits.view(B*T, ND)
#             target_flat = A.view(B*T)
#             mask_flat   = M.view(B*T)
#
#             loss_per_token = ce(logits_flat, target_flat)
#             loss = (loss_per_token * mask_flat).sum() / mask_flat.sum()
#
#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 0.25)
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             writer.add_scalar("train/batch_loss", loss.item(), global_step)
#             global_step += 1
#
#             pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         avg_loss = epoch_loss / len(loader)
#         print(f"Epoch {epoch:3d}/{EPOCHS:3d} — loss {avg_loss:.4f} — time {time.time()-t0:.1f}s")
#         writer.add_scalar("train/epoch_loss", avg_loss, epoch)
#         if epoch%2==0:
#             torch.save(model.state_dict(), f"dt_graph_masked{epoch}.pt")
#
#     writer.close()
#     torch.save(model.state_dict(), "dt_graph_masked.pt")
#     print("✅ Done training.")
#
# if __name__ == "__main__":
#     main()

import pickle
import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from myDataLoader import GraphTrajectoryDataset, collate_fn
from model_graph import DecisionTransformerRCGraph
from config import MAX_LENGTH, BATCH_SIZE, DEVICE

# Hyperparameters
LR        = 1e-4
EPOCHS    = 20
DATA_FILE = "graph_random_10000.pkl"
LAMBDA_RTG_LOSS = 0.1  # how much weight to give to RTG prediction loss

def main():
    # Graph
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(i, i+1) for i in range(9)])
    G.add_edges_from([(0,5), (2,7), (3,8), (1,4)])

    # Load dataset
    with open(DATA_FILE, "rb") as f:
        episodes = pickle.load(f)

    ds = GraphTrajectoryDataset(episodes, G, MAX_LENGTH)
    loader = DataLoader(ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=collate_fn,
                        drop_last=True)

    model = DecisionTransformerRCGraph(
        num_nodes   = G.number_of_nodes(),
        max_length  = MAX_LENGTH,
        scale       = 1.0
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(reduction="none")
    mse = nn.MSELoss()

    writer = SummaryWriter("runs/graph_dt_masked_rtg")
    global_step = 0

    for epoch in range(1, EPOCHS+1):
        t0, epoch_loss = time.time(), 0.0
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for batch in pbar:
            S  = batch["states"].to(DEVICE)         # (B,T)
            A  = batch["actions"].to(DEVICE)        # (B,T)
            R  = batch["returns_to_go"].to(DEVICE)  # (B,T,1)
            Tm = batch["timesteps"].to(DEVICE)      # (B,T)
            VM = batch["valid_mask"].to(DEVICE)     # (B,T,act_dim)
            M  = batch["attention_mask"].to(DEVICE) # (B,T)

            # print(R )

            logits, pred_rtg = model(S, A, R, Tm, VM)
            B,T,ND = logits.shape

            # Action loss (cross entropy)
            logits_flat = logits.view(B*T, ND)
            target_flat = A.view(B*T)
            mask_flat   = M.view(B*T)
            loss_ce = ce(logits_flat, target_flat)
            loss_ce = (loss_ce * mask_flat).sum() / mask_flat.sum()

            # RTG prediction loss
            pred_rtg_flat = pred_rtg.view(B*T, 1)
            true_rtg_flat = R.view(B*T, 1)
            mask_rtg      = M.view(B*T, 1)
            loss_rtg = ((pred_rtg_flat - true_rtg_flat)**2 * mask_rtg).sum() / mask_rtg.sum()

            # Combined loss
            loss = loss_ce + LAMBDA_RTG_LOSS * loss_rtg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/loss_ce", loss_ce.item(), global_step)
            writer.add_scalar("train/loss_rtg", loss_rtg.item(), global_step)
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.4f}", rtg=f"{loss_rtg.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d}/{EPOCHS:3d} — loss {avg_loss:.4f} — time {time.time()-t0:.1f}s")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        if epoch%2==0:
            torch.save(model.state_dict(), f"dt_graph_masked_rtg{epoch}.pt")

    writer.close()
    torch.save(model.state_dict(), "dt_graph_masked_rtg.pt")
    print("✅ Done training with RTG prediction.")

if __name__ == "__main__":
    main()
