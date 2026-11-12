import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Expert: 每个专家拥有独立的特征处理层和分类头
# ============================================================
class Expert(nn.Module):
    def __init__(self, num_classes=10):
        super(Expert, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ModelCNNCifar10(nn.Module):
    def __init__(self):
        super(ModelCNNCifar10, self).__init__()

        self.num_experts = 4
        self.top_k = 2
        self.num_classes = 10

        # 32-channel, two-layer CNN Backbone: unchanged.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.router = nn.Linear(8 * 8 * 32, self.num_experts)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.router.bias)

        self.experts = nn.ModuleList([
            Expert(num_classes=self.num_classes)
            for _ in range(self.num_experts)
        ])

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        for expert in self.experts:
            expert.fc1.apply(init_weights)

        self.last_expert_counts = torch.zeros(
            self.num_experts, dtype=torch.long
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.conv(x)
        features = features.view(batch_size, -1)
        router_logits = self.router(features)

        # The original Top-2 sparse routing operation is unchanged.
        topk_logits, topk_indices = torch.topk(
            router_logits, k=self.top_k, dim=1
        )

        self.last_expert_counts = torch.bincount(
            topk_indices.detach().reshape(-1),
            minlength=self.num_experts
        )

        # Normalize selected routing weights (UNCHANGED inference path).
        topk_weights = F.softmax(topk_logits, dim=1)

        # ========================================================
        # Top-2 Router load-balancing auxiliary loss
        # f_e: fraction of selected Top-2 slots going to expert e.
        # p_e: mean FULL-softmax router probability for expert e.
        # L_balance = E * sum_e f_e * p_e (balanced reference ~1).
        #
        # Hard Top-k selection is non-differentiable, so f_e is constant;
        # p_e stays in the computation graph and gives router gradients.
        # Do NOT use the selected-only Top-2 softmax for p_e.
        # ========================================================
        full_router_probs = F.softmax(router_logits, dim=1)
        hard_load = self.last_expert_counts.to(
            device=router_logits.device, dtype=router_logits.dtype
        ) / float(batch_size * self.top_k)
        mean_router_probs = full_router_probs.mean(dim=0)

        self.last_router_balance_loss = (
            self.num_experts * torch.sum(hard_load.detach() * mean_router_probs)
        )
        # Detached only for CSV monitoring, not for auxiliary loss.
        self.last_router_mean_probs = mean_router_probs.detach()

        output = features.new_zeros(batch_size, self.num_classes)
        for expert_idx, expert in enumerate(self.experts):
            sample_indices, topk_slots = torch.where(
                topk_indices == expert_idx
            )
            if sample_indices.numel() == 0:
                continue

            expert_features = features.index_select(0, sample_indices)
            expert_output = expert(expert_features)
            weights = topk_weights[
                sample_indices, topk_slots
            ].unsqueeze(1)
            output = output.index_add(
                0, sample_indices, expert_output * weights
            )
        return output
