import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def masked_mean(self, x, lengths):
        B, T, D = x.shape
        device = x.device

        mask = torch.arange(T, device=device).expand(B, T)
        mask = mask < lengths.unsqueeze(1)  # [B, T]

        mask = mask.unsqueeze(-1)  # [B, T, 1]
        x = x * mask

        summed = x.sum(dim=1)  # [B, D]
        lengths = lengths.clamp(min=1).unsqueeze(1)

        return summed / lengths  # [B, D]

    def forward(self, x, y, y_lengths):
        # Pool both sequences
        x_repr = x.mean(dim=1)  # no padding assumed for x
        y_repr = self.masked_mean(y, y_lengths)

        # Normalize
        x_repr = F.normalize(x_repr, dim=-1)
        y_repr = F.normalize(y_repr, dim=-1)

        loss = 1 - F.cosine_similarity(x_repr, y_repr, dim=-1)  # [B]

        return loss.mean()