import torch
import torch.nn as nn
from transformers import GPT2Model

class DecisionTransformerRC(nn.Module):

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int = 20,
        transformer_name: str = "gpt2",
        action_tanh: bool = True,
        state_mean: torch.Tensor = None,
        state_std: torch.Tensor = None,
        scale: float = None,
    ):
        super().__init__()
        # GPT-2 backbone
        self.transformer = GPT2Model.from_pretrained(transformer_name)
        # freeze positional embeddings
        self.transformer.wpe.weight.data.zero_()
        self.transformer.wpe.weight.requires_grad = False

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = self.transformer.config.n_embd
        self.max_length = max_length

        # normalization buffers
        if state_mean is not None and state_std is not None:
            self.register_buffer("state_mean", state_mean.view(1,1,-1).float())
            self.register_buffer("state_std",  state_std.view(1,1,-1).float())
        else:
            self.state_mean = None
            self.state_std = None
        self.scale = scale

        H = self.hidden_size
        # embeddings
        self.embed_timestep = nn.Embedding(max_length, H)
        self.embed_return = nn.Sequential(
            nn.Linear(1, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.Tanh()
        )
        self.embed_state = nn.Linear(state_dim, H)
        self.embed_action = nn.Sequential(
            nn.Linear(act_dim, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.Tanh()
        )
        self.embed_ln = nn.LayerNorm(H)

        # heads
        self.predict_return = nn.Linear(H, 1)
        self.predict_state = nn.Linear(H, state_dim)
        act_head = [nn.Linear(H, act_dim)]
        if action_tanh:
            act_head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*act_head)

    def forward(self, states, actions, returns_to_go, timesteps):
        B, T = states.shape[:2]
        H = self.hidden_size
        # normalize states
        if self.state_mean is not None:
            states = (states - self.state_mean) / self.state_std
        # scale RTG
        if self.scale is not None:
            returns_to_go = returns_to_go / self.scale

        # embeddings
        t_emb = self.embed_timestep(timesteps)       # (B,T,H)
        r_emb = self.embed_return(returns_to_go)     # (B,T,H)
        s_emb = self.embed_state(states)             # (B,T,H)
        a_emb = self.embed_action(actions)           # (B,T,H)
        # add timestep
        r_emb = r_emb + t_emb
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # interleave [R,S,A]
        x = torch.stack((r_emb, s_emb, a_emb), dim=2).view(B, 3*T, H)
        x = self.embed_ln(x)
        attn_mask = torch.ones(B, 3*T, device=x.device, dtype=torch.bool)
        out = self.transformer(inputs_embeds=x,
                               attention_mask=attn_mask,
                               use_cache=False)
        # reshape to (B,3,T,H)
        h = out.last_hidden_state.view(B, T, 3, H).permute(0,2,1,3)

        # predictions
        return_preds = self.predict_return(h[:,2])   # (B,T,1)
        state_preds  = self.predict_state(h[:,2])    # (B,T,state_dim)
        action_preds = self.predict_action(h[:,1])   # (B,T,act_dim)
        return state_preds, action_preds, return_preds

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps):
        # states: (T,state_dim), actions: (T,act_dim), returns_to_go: (T,1), timesteps: (T,)
        # add batch dim
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        returns_to_go = returns_to_go.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        # pad/crop
        B, T, _ = states.shape
        if T > self.max_length:
            states, actions, returns_to_go, timesteps = [x[:, -self.max_length:] for x in (states, actions, returns_to_go, timesteps)]
        else:
            pad = self.max_length - T
            pad_s = torch.zeros(1, pad, self.state_dim, device=states.device)
            pad_a = torch.zeros(1, pad, self.act_dim, device=actions.device)
            pad_r = torch.zeros(1, pad, 1, device=returns_to_go.device)
            pad_t = torch.zeros(1, pad, dtype=torch.long, device=timesteps.device)
            states = torch.cat([pad_s, states], dim=1)
            actions = torch.cat([pad_a, actions], dim=1)
            returns_to_go = torch.cat([pad_r, returns_to_go], dim=1)
            timesteps = torch.cat([pad_t, timesteps], dim=1)
        # rebuild timesteps
        B, T, _ = states.shape
        timesteps = torch.arange(T, device=states.device).unsqueeze(0)
        # forward and pick last action
        _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1]
