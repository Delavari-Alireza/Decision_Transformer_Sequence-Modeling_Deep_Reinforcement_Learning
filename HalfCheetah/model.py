import torch
import torch.nn as nn
from transformers import GPT2Model

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim:    int,
        act_dim:      int,
        max_length:   int = 20,
        transformer_name: str = "gpt2",
        action_tanh:  bool = True,
        state_mean:   torch.Tensor = None,
        state_std:    torch.Tensor = None,
        scale:        float = None,
    ):
        super().__init__()
        # 1) GPT-2 backbone
        self.transformer = GPT2Model.from_pretrained(transformer_name)
        # zero & freeze its positional embeddings
        self.transformer.wpe.weight.data.zero_()
        self.transformer.wpe.weight.requires_grad = False

        # 2) dims & normalization constants
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = self.transformer.config.n_embd
        self.max_length  = max_length

        if state_mean is not None and state_std is not None:
            # register buffers so they move with .to(device)
            self.register_buffer("state_mean", state_mean.view(1,1,-1).float())
            self.register_buffer("state_std",  state_std.view(1,1,-1).float())
        else:
            self.state_mean = None
            self.state_std  = None

        # scale for returns-to-go
        self.scale = scale

        # 3) modality & timestep embeddings
        H = self.hidden_size
        self.embed_timestep = nn.Embedding(max_length, H)
        self.embed_return   = nn.Linear(1, H)
        self.embed_state    = nn.Linear(state_dim, H)
        self.embed_action   = nn.Linear(act_dim,   H)
        self.embed_ln       = nn.LayerNorm(H)

        # 4) prediction heads
        self.predict_return = nn.Linear(H, 1)
        self.predict_state  = nn.Linear(H, state_dim)
        act_head = [nn.Linear(H, act_dim)]
        if action_tanh:
            act_head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*act_head)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        states:        (B, T, state_dim) raw states
        actions:       (B, T, act_dim)
        returns_to_go: (B, T, 1) raw (unscaled) returns
        timesteps:     (B, T) long ints
        """
        B, T = states.shape[:2]

        # — normalize if provided —
        if self.state_mean is not None:
            states = (states - self.state_mean) / self.state_std
        if self.scale is not None:
            returns_to_go = returns_to_go / self.scale

        # — embed each modality + timestep —
        t_emb = self.embed_timestep(timesteps)             # (B,T,H)
        r_emb = self.embed_return(returns_to_go) + t_emb   # (B,T,H)
        s_emb = self.embed_state(states)   + t_emb         # (B,T,H)
        a_emb = self.embed_action(actions) + t_emb         # (B,T,H)

        # — interleave into (B,3T,H) and LN —
        x = torch.stack((r_emb, s_emb, a_emb), dim=2).view(B, 3*T, self.hidden_size)
        x = self.embed_ln(x)

        # — causal attention mask (all ones, bool) —
        attn_mask = torch.ones(B, 3*T, device=x.device, dtype=torch.bool)

        # — feed through GPT-2 (no KV cache) —
        out = self.transformer(
            inputs_embeds  = x,
            attention_mask = attn_mask,
            use_cache      = False
        )
        h = out.last_hidden_state  # (B,3T,H)

        # — reshape back to (B,3,T,H) and select tokens —
        h = h.view(B, T, 3, self.hidden_size).permute(0,2,1,3)
        #   h[:,0] = returns, h[:,1] = states, h[:,2] = actions

        # — prediction heads —
        return_preds = self.predict_return(h[:,2])   # (B,T,1)
        state_preds  = self.predict_state(h[:,2])    # (B,T,state_dim)
        action_preds = self.predict_action(h[:,1])   # (B,T,act_dim)

        return state_preds, action_preds, return_preds


    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps):
        # 1) add batch dim
        states        = states.unsqueeze(0)        # (1, T, S)
        actions       = actions.unsqueeze(0)       # (1, T, A)
        returns_to_go = returns_to_go.unsqueeze(0) # (1, T, 1)
        timesteps     = timesteps.unsqueeze(0)     # (1, T)

        # 2) crop or left-pad to max_length
        L = self.max_length
        if states.shape[1] > L:
            states, actions, returns_to_go, timesteps = [
                x[:, -L:] for x in (states, actions, returns_to_go, timesteps)
            ]
        else:
            pad = L - states.shape[1]
            pad_s = torch.zeros(1, pad, self.state_dim,  device=states.device)
            pad_a = torch.zeros(1, pad, self.act_dim,    device=actions.device)
            pad_r = torch.zeros(1, pad, 1,               device=returns_to_go.device)
            pad_t = torch.zeros(1, pad, dtype=torch.long,device=timesteps.device)
            states        = torch.cat([pad_s, states],        dim=1)
            actions       = torch.cat([pad_a, actions],       dim=1)
            returns_to_go = torch.cat([pad_r, returns_to_go], dim=1)
            # we discard the old timestep values...
            timesteps     = torch.cat([pad_t, timesteps],     dim=1)

        # 3) re-build timesteps
        B, T, _ = states.shape
        timesteps = (
            torch.arange(T, device=states.device, dtype=torch.long)
                 .unsqueeze(0).expand(B, -1)
        )  # shape (B, T)

        # 4) forward + grab last action
        _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1]
