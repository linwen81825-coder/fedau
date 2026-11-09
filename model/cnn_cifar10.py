
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

        # 每个专家独立的分类头
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ============================================================
# CIFAR10 MoE: 4 Experts + Top-2 Routing
# ============================================================

class ModelCNNCifar10(nn.Module):

    def __init__(self):
        super(ModelCNNCifar10, self).__init__()

        self.num_experts = 4
        self.top_k = 2
        self.num_classes = 10

        # ----------------------------------------------------
        # 1. Shared CNN Backbone
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 2. Router
        # ----------------------------------------------------

        self.router = nn.Linear(
            8 * 8 * 32,
            self.num_experts
        )

        # Router 使用较小的初始化，避免初始路由分数过大
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.router.bias)

        # ----------------------------------------------------
        # 3. Four Independent Experts
        # ----------------------------------------------------

        self.experts = nn.ModuleList([
            Expert(num_classes=self.num_classes)
            for _ in range(self.num_experts)
        ])

        # ----------------------------------------------------
        # 4. Weight Initialization
        # ----------------------------------------------------

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Shared CNN initialization
        self.conv.apply(init_weights)

        # Initialize expert feature extraction layers
        for expert in self.experts:
            expert.fc1.apply(init_weights)

        # 专家分类头保留 PyTorch 默认初始化

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        batch_size = x.size(0)

        # ----------------------------------------------------
        # 1. Shared Feature Extraction
        # ----------------------------------------------------

        features = self.conv(x)

        features = features.view(batch_size, -1)

        # ----------------------------------------------------
        # 2. Router
        # ----------------------------------------------------

        router_logits = self.router(features)

        # Select Top-2 experts for each sample
        topk_logits, topk_indices = torch.topk(
            router_logits,
            k=self.top_k,
            dim=1
        )

        # Normalize selected routing weights
        topk_weights = F.softmax(topk_logits, dim=1)

        # ----------------------------------------------------
        # 3. Sparse Expert Computation
        # ----------------------------------------------------

        output = features.new_zeros(
            batch_size,
            self.num_classes
        )

        for expert_idx, expert in enumerate(self.experts):

            # Find samples assigned to this expert
            sample_indices, topk_slots = torch.where(
                topk_indices == expert_idx
            )

            if sample_indices.numel() == 0:
                continue

            # Extract corresponding features
            expert_features = features.index_select(
                0,
                sample_indices
            )

            # Expert-specific classification
            expert_output = expert(expert_features)

            # Corresponding Top-2 routing weights
            weights = topk_weights[
                sample_indices,
                topk_slots
            ].unsqueeze(1)

            # Weighted aggregation of expert logits
            output = output.index_add(
                0,
                sample_indices,
                expert_output * weights
            )

        return output
