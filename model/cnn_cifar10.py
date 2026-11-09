
import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """
    一个完整的专家：
    1. 专家特征变换网络
    2. 专家独立分类头
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=128,
        num_classes=10
    ):
        super(ExpertMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(self, x):

        features = self.net(x)

        logits = self.classifier(features)

        return logits


class ModelCNNCifar10(nn.Module):

    def __init__(self, num_experts=4):

        super(ModelCNNCifar10, self).__init__()

        self.num_experts = num_experts

        if num_experts < 2:
            raise ValueError("Top-2路由至少需要两个专家")

        # =====================================
        # 1. 共享 CNN 特征提取层
        # =====================================

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # =====================================
        # 2. 共享特征映射层
        # =====================================

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # =====================================
        # 3. Router
        # =====================================

        self.router = nn.Linear(64, num_experts)

        # =====================================
        # 4. 多个专家
        # 每个专家都有独立分类头
        # =====================================

        self.experts = nn.ModuleList([
            ExpertMLP(
                input_dim=64,
                hidden_dim=128,
                num_classes=10
            )
            for _ in range(num_experts)
        ])

        # 辅助信息
        self.aux_loss = None
        self.last_expert_indices = None
        self.last_expert_gates = None

        # =====================================
        # 5. 参数初始化
        # =====================================

        def init_weights(m):

            if isinstance(m, (nn.Linear, nn.Conv2d)):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity='relu'
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):

        # =====================================
        # Step 1: 共享 CNN 提取特征
        # =====================================

        conv_features = self.conv(x)

        conv_features = conv_features.view(
            -1, 8 * 8 * 32
        )

        features = self.fc1(conv_features)

        # =====================================
        # Step 2: Router 计算路由概率
        # =====================================

        router_logits = self.router(features)

        router_probs = torch.softmax(
            router_logits,
            dim=1
        )

        # =====================================
        # Step 3: Top-2 路由
        # =====================================

        # 选择分数最高的两个专家
        # top2_logits 保留梯度
        # top2_indices 只负责选择专家

        top2_logits, top2_indices = torch.topk(
            router_logits,
            k=2,
            dim=1
        )

        # 对选中的两个专家重新归一化
        # 这里的权重可以参与反向传播

        top2_gates = torch.softmax(
            top2_logits,
            dim=1
        )

        # =====================================
        # Step 4: 两个专家分别计算
        # =====================================

        batch_size = features.size(0)

        num_classes = 10

        output = features.new_zeros(
            batch_size,
            num_classes
        )

        for expert_id, expert in enumerate(self.experts):

            # 找出选择当前专家的样本及其Top-2位置
            expert_mask = top2_indices.eq(expert_id)

            sample_indices, gate_positions = torch.nonzero(
                expert_mask,
                as_tuple=True
            )

            if sample_indices.numel() == 0:
                continue

            # 获取当前专家对应的输入
            expert_input = features.index_select(
                0,
                sample_indices
            )

            # 专家独立分类
            expert_logits = expert(expert_input)

            # 获取当前专家对应的Router权重
            expert_gates = top2_gates[
                sample_indices,
                gate_positions
            ].unsqueeze(1)

            # Router权重与专家输出相乘
            weighted_logits = (
                expert_logits * expert_gates
            )

            # 使用index_add进行累加
            # 每个样本会接收两个专家的输出
            # 该操作保留计算图，可以反向传播

            output = output.index_add(
                0,
                sample_indices,
                weighted_logits
            )

        # =====================================
        # Step 5: Router负载均衡辅助损失
        # =====================================

        # 所有专家的平均路由概率
        importance = router_probs.mean(dim=0)

        # Top-2专家实际接收的任务数量
        load = torch.bincount(
            top2_indices.reshape(-1),
            minlength=self.num_experts
        ).to(features.dtype) / (batch_size * 2)

        # 负载均衡损失
        self.aux_loss = self.num_experts * torch.sum(
            importance * load
        )

        # =====================================
        # Step 6: 保存专家分配情况
        # =====================================

        # 形状: [batch_size, 2]
        self.last_expert_indices = top2_indices.detach()

        self.last_expert_gates = top2_gates.detach()

        return output
