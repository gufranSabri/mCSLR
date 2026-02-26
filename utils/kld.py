import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceKLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, lgt):
        B, T, D = x.shape

        log_probs_x = F.log_softmax(x, dim=-1)   # (B, T, D)
        probs_y = F.softmax(y, dim=-1)          # (B, T, D)

        # KL per element (no reduction)
        kl = F.kl_div(
            log_probs_x,
            probs_y,
            reduction="none"
        )  # (B, T, D)

        # Sum over distribution dimension
        kl = kl.sum(dim=-1)  # (B, T)

        # Create mask
        device = x.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lgt.unsqueeze(1)
        # mask shape: (B, T)

        # Apply mask
        kl = kl * mask

        # Normalize by number of valid tokens
        total_valid = mask.sum()
        loss = kl.sum() / total_valid.clamp(min=1)

        return loss