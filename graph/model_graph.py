import torch
import torch.nn as nn
from transformers import GPT2Model
from config import MAX_LENGTH , DEVICE
class DecisionTransformerRCGraph(nn.Module):
    def __init__(self, num_nodes, max_length=MAX_LENGTH, transformer_name="gpt2", scale=1.0):
        super().__init__()
        # 1) load GPT-2 backbone
        self.transformer = GPT2Model.from_pretrained(transformer_name)
        # zero & freeze its positional embeddings
        self.transformer.wpe.weight.data.zero_()
        self.transformer.wpe.weight.requires_grad = False

        self.act_dim     = num_nodes
        self.hidden_size = self.transformer.config.n_embd
        self.max_length  = max_length
        self.scale       = scale

        H = self.hidden_size
        L = self.max_length

        # 2) modality embeddings
        self.embed_timestep = nn.Embedding(L, H)
        self.embed_return   = nn.Linear(1, H) #nn.Sequential(nn.Linear(1, H), nn.Tanh())
        self.embed_state    = nn.Embedding(num_nodes, H)
        self.embed_action   = nn.Sequential(nn.Embedding(num_nodes, H), nn.Tanh())
        self.embed_ln       = nn.LayerNorm(H)

        # 3) prediction heads
        self.predict_return = nn.Linear(H, 1)
        self.predict_action = nn.Linear(H, num_nodes)

    def forward(self, states, actions, returns_to_go, timesteps, valid_mask=None):
        B, T = states.shape
        H = self.hidden_size

        # Modality embeddings
        t_emb = self.embed_timestep(timesteps)
        r_emb = self.embed_return(returns_to_go / self.scale) + t_emb
        s_emb = self.embed_state(states) + t_emb
        a_emb = self.embed_action(actions) + t_emb

        x = torch.stack((r_emb, s_emb, a_emb), dim=2).reshape(B, 3 * T, H)
        x = self.embed_ln(x)

        attn_mask = torch.ones(B, 3 * T, dtype=torch.bool, device=x.device)

        out = self.transformer(
            inputs_embeds=x,
            attention_mask=attn_mask,
            use_cache=False
        )
        h = out.last_hidden_state.view(B, T, 3, H).permute(0, 2, 1, 3)  # (B, 3, T, H)

        act_logits = self.predict_action(h[:, 1])  # from state tokens
        pred_rtg = self.predict_return(h[:, 1])  # also from state tokens

        if valid_mask is not None:
            neg_inf = torch.finfo(act_logits.dtype).min
            act_logits = torch.where(valid_mask, act_logits, neg_inf)

        return act_logits, pred_rtg

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps, valid_actions):
        """
        single‐episode unbatched inference:
        states:        (T,)   long
        actions:       (T,)   long
        returns_to_go: (T,1)  float
        timesteps:     (T,)   long
        valid_actions: list[int] of neighbors at current node
        """
        # 1) batchify + pad/crop to max_length
        S  = states.unsqueeze(0)
        A  = actions.unsqueeze(0)
        R  = returns_to_go.unsqueeze(0)
        Tm = timesteps.unsqueeze(0)
        T  = S.shape[1]
        L  = self.max_length

        if T > L:
            S, A, R, Tm = [x[:, -L:] for x in (S, A, R, Tm)]
        else:
            pad = L - T
            S   = torch.cat([torch.zeros(1,pad,dtype=torch.long, device=DEVICE), S],  dim=1)
            A   = torch.cat([torch.zeros(1,pad,dtype=torch.long, device=DEVICE), A],  dim=1)
            R   = torch.cat([torch.zeros(1,pad,1,               device=DEVICE), R],  dim=1)
            Tm  = torch.cat([torch.zeros(1,pad,dtype=torch.long, device=DEVICE), Tm], dim=1)

        # 2) rebuild timesteps 0…L-1
        Tm = torch.arange(L, device=DEVICE).unsqueeze(0)

        # 3) build valid‐action mask for last position, excluding “stay”
        vm = torch.zeros(1, L, self.act_dim, dtype=torch.bool, device=DEVICE)
        # only allow true neighbors, never the current node itself
        for a in valid_actions:
            if a != int(states[-1]):
                vm[0, -1, a] = True

        # 4) forward + pick best
        logits = self.forward(S.to(DEVICE), A.to(DEVICE), R.to(DEVICE), Tm, vm)
        return int(logits[0, -1].argmax().item())
    @torch.no_grad()
    def get_action(self, state_hist, action_hist, rtg_hist, timestep, target_rtg, reward, **kwargs):
        """
        Generate next action using Decision Transformer.

        state_hist   : list of past states (ints), len <= max_length
        action_hist  : list of past actions (ints), len == len(state_hist)
        rtg_hist     : list of past RTG values, len == len(state_hist)
        timestep     : int, current environment timestep
        target_rtg   : float, desired return-to-go for this episode
        reward       : float, most recent reward from environment

        Returns: next action (int)
        """
        device = next(self.parameters()).device
        T = self.max_length

        # Step 1: update return-to-go history
        target_rtg -= reward
        rtg_hist = rtg_hist[1:] + [target_rtg] if len(rtg_hist) == T else rtg_hist + [target_rtg]

        # Step 2: truncate histories
        state_hist = state_hist[-T:]
        action_hist = action_hist[-T:]
        rtg_hist = rtg_hist[-T:]

        # Step 3: pad histories if needed
        pad_len = T - len(state_hist)
        state_pad = [0] * pad_len
        action_pad = [0] * pad_len
        rtg_pad = [0.0] * pad_len

        states = torch.tensor(state_pad + state_hist, dtype=torch.long, device=device).unsqueeze(0)
        actions = torch.tensor(action_pad + action_hist, dtype=torch.long, device=device).unsqueeze(0)
        rtgs = torch.tensor(rtg_pad + rtg_hist, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        timesteps = torch.arange(len(state_pad + state_hist), device=device).unsqueeze(0)

        # Step 4: attention mask (0 for pad, 1 for real)
        attention_mask = torch.tensor(
            [0] * pad_len + [1] * len(state_hist), dtype=torch.long, device=device
        ).unsqueeze(0)

        # Step 5: forward through model
        _, action_preds, _ = self.forward(
            states, actions, None, rtgs, timesteps, attention_mask=attention_mask, **kwargs
        )

        action = action_preds[0, -1].argmax(dim=-1).item() if action_preds.shape[-1] > 1 else action_preds[0, -1].item()
        return action, rtg_hist, target_rtg
