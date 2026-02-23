import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components.utils import PositionwiseFeedForward


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_dim),
        )
    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        assert self.k <= self.num_experts, "k must be <= num_experts"
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)  # [B, E]
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(scores, self.k, dim=-1) # [B, k]

        dispatch_mask = torch.zeros_like(scores)
        dispatch_mask.scatter_(1, topk_indices, 1.0)

        load = dispatch_mask.sum(dim=0) # [E]
        importance = scores.sum(dim=0) # [E]
        B, E = scores.shape
        aux_loss = (importance * load).sum() * self.num_experts / (B ** 2)

        return dispatch_mask, scores, aux_loss, topk_indices

class MoE(nn.Module):
    def __init__(self, input_dim, ff_size, ff_kernelsize,
                 num_experts, top_k):
        super().__init__()

        self.router = Router(input_dim, num_experts, top_k)

        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(
                    input_size=input_dim,
                    ff_size=ff_size,
                    dropout=0.1,
                    kernel_size=ff_kernelsize,
                    skip_connection=True
                )
                for _ in range(num_experts)
            ]
        )

        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

    def forward(self, x):
        """
        x: [B, T, D]
        """

        B, T, D = x.shape
        assert D == self.input_dim

        # --------------------------------------------------
        # Flatten tokens so router works per-token
        # --------------------------------------------------
        x_flat = x.reshape(B * T, D)   # [B*T, D]

        dispatch_mask, scores, aux_loss, topk_indices = self.router(x_flat)
        # scores: [B*T, E]
        # topk_indices: [B*T, k]

        batch_tokens = B * T
        batch_times_top_k = batch_tokens * self.top_k

        # --------------------------------------------------
        # Expand inputs for top-k routing
        # --------------------------------------------------
        x_expanded = x_flat.unsqueeze(1).expand(batch_tokens, self.top_k, D)
        topk_scores = torch.gather(scores, 1, topk_indices)

        flat_inputs = x_expanded.reshape(batch_times_top_k, D)
        flat_scores = topk_scores.reshape(batch_times_top_k, 1)
        flat_expert_ids = topk_indices.reshape(batch_times_top_k)

        all_outputs = torch.zeros_like(flat_inputs)

        # --------------------------------------------------
        # Route to experts
        # --------------------------------------------------
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_expert_ids == i)
            if expert_mask.any():
                x_i = flat_inputs[expert_mask]

                # Expert expects [B, T, D]
                # So we fake T=1 for token-level processing
                x_i = x_i.unsqueeze(1)      # [N_i, 1, D]
                y_i = expert(x_i)           # [N_i, 1, D]
                y_i = y_i.squeeze(1)        # [N_i, D]

                all_outputs[expert_mask] = y_i

        # Weight by routing scores
        all_outputs *= flat_scores

        # --------------------------------------------------
        # Combine top-k outputs
        # --------------------------------------------------
        expert_outputs = all_outputs.view(batch_tokens, self.top_k, D).sum(dim=1)

        # Restore [B, T, D]
        expert_outputs = expert_outputs.view(B, T, D)

        return expert_outputs, aux_loss


if __name__ == "__main__":
    B, T, D = 4, 10, 16
    num_experts = 4
    top_k = 2
    ff_size = 64
    ff_kernelsize = 3
    num_classes = 20

    x = torch.randn(B, T, D)
    moe_layer = MoE(input_dim=D, ff_size=ff_size, ff_kernelsize=ff_kernelsize,
                    num_experts=num_experts, top_k=top_k, num_classes=num_classes)
    logits, aux_loss, dispatch_mask, scores = moe_layer(x)

    print("Logits shape:", logits.shape)  # [B, T, num_classes]
    print("Aux loss:", aux_loss.item())