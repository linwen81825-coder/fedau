
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
# Residual CNN Block
# 每个专家拥有独立的 Residual CNN + GroupNorm
# 输入:  [B, 32, 8, 8]
# 输出:  [B, 32, 8, 8]
# ============================================================

class ResidualConvBlock(nn.Module):

    def __init__(self, channels=32, groups=8):

        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.gn1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=channels
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.gn2 = nn.GroupNorm(
            num_groups=groups,
            num_channels=channels
        )

        # 卷积层初始化
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode='fan_out',
            nonlinearity='relu'
        )

        nn.init.kaiming_normal_(
            self.conv2.weight,
            mode='fan_out',
            nonlinearity='relu'
        )

        # 残差分支采用较小的初始权重
        nn.init.constant_(self.gn2.weight, 0.1)
        nn.init.zeros_(self.gn2.bias)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        # Residual Connection
        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# Expert
# Residual CNN + GroupNorm + MLP + Independent Classifier
# ============================================================

class Expert(nn.Module):

    def __init__(self, num_classes=10):

        super(Expert, self).__init__()

        # ----------------------------------------------------
        # 1. Independent Residual CNN
        # ----------------------------------------------------

        self.feature_extractor = ResidualConvBlock(
            channels=32,
            groups=8
        )

        # ----------------------------------------------------
        # 2. Original MLP
        # ----------------------------------------------------

        self.fc1 = nn.Sequential(

            nn.Linear(8 * 8 * 32, 256),

            nn.ReLU(),

            nn.Linear(256, 64),

            nn.ReLU()
        )

        # ----------------------------------------------------
        # 3. Independent Classification Head
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        # Independent CNN feature extraction
        x = self.feature_extractor(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # MLP
        x = self.fc1(x)

        # Independent classifier
        x = self.fc2(x)

        return x


# ============================================================
# CIFAR10 MoE
# Shared CNN Backbone + Router + 4 Experts + Top-2
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
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # ----------------------------------------------------
        # 2. Router
        # Router input: Shared Backbone features
        # ----------------------------------------------------

        self.router = nn.Linear(
            8 * 8 * 32,
            self.num_experts
        )

        nn.init.normal_(
            self.router.weight,
            mean=0.0,
            std=0.001
        )

        nn.init.zeros_(
            self.router.bias
        )

        # ----------------------------------------------------
        # 3. Four Independent Experts
        # ----------------------------------------------------

        self.experts = nn.ModuleList([

            Expert(
                num_classes=self.num_classes
            )

            for _ in range(self.num_experts)

        ])

        # ----------------------------------------------------
        # 4. Initialization
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

        # Initialize shared Backbone
        self.conv.apply(init_weights)

        # Initialize expert MLPs
        for expert in self.experts:

            expert.fc1.apply(init_weights)

        # Expert residual CNNs have already been initialized
        # independently in ResidualConvBlock.
        # Expert classification heads retain PyTorch defaults.

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        batch_size = x.size(0)

        # ----------------------------------------------------
        # 1. Shared Backbone
        # ----------------------------------------------------

        feature_map = self.conv(x)

        # [B, 32, 8, 8] -> [B, 2048]
        features = feature_map.view(
            batch_size,
            -1
        )

        # ----------------------------------------------------
        # 2. Router
        # ----------------------------------------------------

        router_logits = self.router(features)

        # Top-2 routing
        topk_logits, topk_indices = torch.topk(
            router_logits,
            k=self.top_k,
            dim=1
        )

        # Normalize Top-2 weights
        topk_weights = F.softmax(
            topk_logits,
            dim=1
        )

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

            # Extract shared feature maps
            expert_features = feature_map.index_select(
                0,
                sample_indices
            )

            # Independent Residual CNN + MLP + Classifier
            expert_output = expert(expert_features)

            # Corresponding routing weights
            weights = topk_weights[
                sample_indices,
                topk_slots
            ].unsqueeze(1)

            # Weighted aggregation
            output = output.index_add(
                0,
                sample_indices,
                expert_output * weights
            )

        return output
