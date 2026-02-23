import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceAlignmentLoss(nn.Module):
    """
    Aligns two variable-length sequences without matching time steps.
    Uses masked mean pooling + cosine similarity.
    """

    def __init__(self):
        super().__init__()

    def masked_mean(self, x, lengths):
        """
        x: [B, T, D]
        lengths: list or tensor of length B
        """
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
        """
        x: [B, T, D]
        y: [B, T', D]  (already projected to same D)
        y_lengths: [B]
        """

        # Pool both sequences
        x_repr = x.mean(dim=1)  # no padding assumed for x
        y_repr = self.masked_mean(y, y_lengths)

        # Normalize
        x_repr = F.normalize(x_repr, dim=-1)
        y_repr = F.normalize(y_repr, dim=-1)

        # Cosine similarity loss
        loss = 1 - (x_repr * y_repr).sum(dim=-1)

        return loss.mean()