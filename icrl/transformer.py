import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. fixed sinusoidal positions (parameter-free)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        # B - batch size
        # T - sequence length
        # d - embedding dimension
        return self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)  # fused W^Q|W^K|W^V
        self.out = nn.Linear(d_model, d_model)  # W^O
        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,D)
        causal_mask: Optional[torch.Tensor] = None,  # (T,T) with -inf above diag
        key_padding_mask: Optional[torch.Tensor] = None,  # (B,T) True where PAD
    ):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Q,K,V
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)  # each (B,T,D)

        def split_heads(t):  # (B,T,D) -> (B,H,T,Dh)
            return t.view(B, T, H, Dh).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)  # (B,H,T,Dh)

        # Attention scores QK^T / sqrt(Dh)
        # Tq is the length of q (usually T); Tk is current key length (T or Ttot if cached)
        Tq, Tk = q.size(2), k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            Dh
        )  # (B,H,Tq,Tk)

        # Causal mask (only needed when we computed a full T-long block of keys this call)
        if causal_mask is not None:
            # If using cache for single-step decode, you can pass None here;
            # otherwise provide (T,T) and slice to (Tq,Tk) if shapes differ.
            if causal_mask.size(0) != Tq or causal_mask.size(1) != Tk:
                cm = causal_mask[-Tq:, -Tk:]  # align to current window
            else:
                cm = causal_mask
            attn_scores = attn_scores + cm

        # Key padding mask: True where PAD -> -inf
        if key_padding_mask is not None:
            # Expect mask over keys; if using cache, mask must match Tk (pad at the left if you prepend)
            kpm = key_padding_mask[:, None, None, :]  # (B,1,1,Tk)
            attn_scores = attn_scores.masked_fill(kpm, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, v)  # (B,H,Tq,Dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, D)  # (B,Tq,D)

        out = self.out(ctx)  # (B,Tq,D)
        return self.drop(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
    ):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, activation="gelu")

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        h = self.attn(h, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = x + h
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


def make_causal_mask(T, device, dtype):
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)  # -inf above diagonal, 0 elsewhere
    return m


def make_key_padding_mask(seq_lens, max_len=None, device=None):
    """
    Create a key padding mask from sequence lengths.

    Args:
        seq_lens (list[int] or torch.Tensor): lengths of each sequence in batch (B,)
        max_len (int, optional): maximum sequence length.
            If None, uses max(seq_lens).
        device: torch device (optional).

    Returns:
        torch.BoolTensor of shape (B, max_len)
        True = PAD position, False = real token.
    """
    if not torch.is_tensor(seq_lens):
        seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=device)
    else:
        seq_lens = seq_lens.to(device)

    if max_len is None:
        max_len = seq_lens.max().item()

    # shape (B, max_len), True where idx >= seq_len
    idxs = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    mask = idxs >= seq_lens.unsqueeze(1)  # (B, max_len)
    return mask


class RLTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rew_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        action_space: str = "discrete",
        num_actions: int | None = None,
        interleave: bool = False,
    ):
        super().__init__()
        self.interleave = interleave
        self.d_model = d_model
        self.input_dim = obs_dim + act_dim + rew_dim

        # 1) feature projection
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Field-wise projections for interleaved tokenization
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        self.rew_proj = nn.Linear(rew_dim, d_model)
        # Token-type embedding: 0=OBS, 1=ACT_PREV, 2=REW_PREV
        self.type_emb = nn.Embedding(3, d_model)

        # 2) sinusoidal positions
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)

        # 3) transformer stack
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # 4) action head
        self.action_space = action_space
        if action_space == "discrete":
            assert num_actions is not None, (
                "num_actions must be set for discrete actions."
            )
            self.head = nn.Linear(d_model, num_actions)
        elif action_space == "continuous":
            self.mu = nn.Linear(d_model, act_dim)
            self.log_std = nn.Linear(d_model, act_dim)
            self.min_log_std, self.max_log_std = -5.0, 2.0
        else:
            raise ValueError("action_space must be 'discrete' or 'continuous'.")

    def forward(self, obs_flat, acts_prev, rews_prev, pad_mask=None):
        """
        obs_flat: (B,T,obs_dim)  already flattened
        acts_prev: (B,T,act_dim) contains a_{t-1}
        rews_prev: (B,T,rew_dim) contains r_{t-1}
        pad_mask: (B,T) bool, True where PAD (optional)

        When interleave=True, the model builds a sequence of length 3*T with tokens ordered
        [O_t, A_{t-1}, R_{t-1}], applies positions over that longer sequence, and produces logits
        only at observation token positions so that the objective matches log p(a_t | h_{t-1}, o_t).
        """
        if not self.interleave:
            # ---- single-token-per-step (default) ----
            x = torch.cat([obs_flat, acts_prev, rews_prev], dim=-1)  # (B,T,input_dim)
            x = self.input_proj(x)  # (B,T,d_model)
            x = x + self.pos_enc(x)  # add positions
            x = self.drop(x)
            seq_pad_mask = pad_mask  # (B,T) or None
        else:
            # ---- interleaved tokenization: [O_t, A_{t-1}, R_{t-1}] ----
            B, T, _ = obs_flat.shape
            # Project each field to d_model
            o_tok = self.obs_proj(obs_flat)  # (B,T,d)
            a_tok = self.act_proj(acts_prev)  # (B,T,d)
            r_tok = self.rew_proj(rews_prev)  # (B,T,d)

            # Add type embeddings
            type_o = self.type_emb.weight[0].view(1, 1, -1)  # (1,1,d)
            type_a = self.type_emb.weight[1].view(1, 1, -1)
            type_r = self.type_emb.weight[2].view(1, 1, -1)
            o_tok = o_tok + type_o
            a_tok = a_tok + type_a
            r_tok = r_tok + type_r

            # Interleave along time: [O_0, A_-1, R_-1, O_1, A_0, R_0, ..., O_{T-1}, A_{T-2}, R_{T-2}]
            # Practically: stack then reshape to (B, 3T, d)
            x = torch.stack([o_tok, a_tok, r_tok], dim=3)  # (B,T,d,3)
            x = x.reshape(B, T * 3, -1)  # (B,3T,d)

            # Positional encoding over length 3T
            x = x + self.pos_enc(x)
            x = self.drop(x)

            # Expand key padding mask from (B,T) to (B,3T): replicate each timestep's mask 3 times
            if pad_mask is not None:
                seq_pad_mask = pad_mask.unsqueeze(-1).expand(B, T, 3).reshape(B, T * 3)
            else:
                seq_pad_mask = None

        # causal mask
        T = x.size(1)
        c_mask = make_causal_mask(T, x.device, x.dtype)  # (T,T)

        # transformer stack
        for blk in self.blocks:
            x = blk(x, causal_mask=c_mask, key_padding_mask=seq_pad_mask)

        x = self.ln_f(x)  # (B, L, d) where L=T or L=3T

        if not self.interleave:
            feat = x  # (B,T,d)
        else:
            # Select features at OBS token positions: indices 0,3,6,... in the interleaved sequence
            B, L, d = x.shape
            # number of logical timesteps inferred when interleaved
            T_eff = L // 3
            idx = torch.arange(0, L, 3, device=x.device)  # (T_eff,)
            feat = x.index_select(dim=1, index=idx)  # (B,T_eff,d)

            # Also, if a pad_mask was provided, collapse the (B,3T) mask back to (B,T) by taking the OBS positions
            if seq_pad_mask is not None:
                seq_pad_mask = seq_pad_mask.index_select(dim=1, index=idx)

        if self.action_space == "discrete":
            out = self.head(feat)  # (B,T,num_actions)
        else:
            mu = self.mu(feat)
            log_std = self.log_std(feat).clamp_(self.min_log_std, self.max_log_std)
            out = (mu, log_std)
        return out
