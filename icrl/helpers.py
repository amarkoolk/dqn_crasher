import torch
from typing import List, Dict, Tuple


@torch.no_grad()
def pack_multi_episode_context(
    batch_episodes: List[
        List[Dict]
    ],  # len B; each = list of step dicts with 'state','action','reward'
    T: int,
    device=None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    """
    Concatenate episodes per batch item up to length T, reset prev a/r at ep starts, and pad tail.

    Returns:
        obs       : (B, T, obs_dim)          float
        acts_prev : (B, T, 1)                float  (scalar prev action as a feature; 0 at ep starts)
        rews_prev : (B, T, 1)                float  (0 at ep starts)
        targets   : (B, T)                   int64  (discrete action target for CE)
        pad_mask  : (B, T)                   bool   True where PAD
        obs_dim   : int
        num_actions: int  (max(action)+1 over provided data)
    """
    # Infer dims from first non-empty episode
    first = next(ep for ep in batch_episodes if len(ep) > 0)
    s0 = torch.as_tensor(first[0]["state"])
    obs_dim = int(s0.numel())
    if device is None:
        device = s0.device if s0.is_cuda else torch.device("cpu")

    # quick scan for num_actions
    all_actions = []
    for ep in batch_episodes:
        for step in ep:
            a = int(step["action"])
            all_actions.append(a)
    num_actions = (max(all_actions) + 1) if all_actions else 1

    B = len(batch_episodes)
    obs = torch.zeros(B, T, obs_dim, device=device, dtype=torch.float32)
    acts_prev = torch.zeros(B, T, 1, device=device, dtype=torch.float32)
    rews_prev = torch.zeros(B, T, 1, device=device, dtype=torch.float32)
    targets = torch.zeros(B, T, device=device, dtype=torch.long)
    pad_mask = torch.ones(B, T, device=device, dtype=torch.bool)  # start as PAD=True

    for b, ep in enumerate(batch_episodes):
        # Flatten this episode list into tensors
        if len(ep) == 0:
            continue
        states = torch.stack(
            [
                torch.as_tensor(st["state"], dtype=torch.float32, device=device).view(
                    -1
                )
                for st in ep
            ]
        )
        actions = torch.tensor(
            [int(st["action"]) for st in ep], device=device, dtype=torch.long
        )
        rewards = torch.tensor(
            [float(st["reward"]) for st in ep], device=device, dtype=torch.float32
        ).unsqueeze(-1)

        # Fill as many steps as fit, then stop (one episode per batch item here; you can pass multiple episodes per item if you want)
        L = min(states.size(0), T)
        obs[b, :L] = states[:L]
        targets[b, :L] = actions[:L]
        pad_mask[b, :L] = False

        # Teacher forcing: prev action/reward shifted within the segment; reset at position 0
        if L > 0:
            if L > 1:
                acts_prev[b, 1:L, 0] = actions[: L - 1].to(torch.float32)
                rews_prev[b, 1:L, 0] = rewards[: L - 1, 0]
            acts_prev[b, 0, 0] = 0.0
            rews_prev[b, 0, 0] = 0.0

    return obs, acts_prev, rews_prev, targets, pad_mask, obs_dim, num_actions
