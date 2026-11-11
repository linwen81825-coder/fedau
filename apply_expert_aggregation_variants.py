#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate five independent expert-aggregation experiment branches.

Every branch is created directly from:
dc2108d21dad3c411b51e3238b1630f6f78028d0

Run this script from the root of the fedau Git repository:
    python apply_expert_aggregation_variants.py

The script requires a clean working tree and a local copy of the base commit.
It creates local branches and commits the three modified files on each branch.
It does not push branches to GitHub.
"""

from pathlib import Path
import subprocess
import sys

BASE = "dc2108d21dad3c411b51e3238b1630f6f78028d0"

VARIANTS = [
    {
        "branch": "expert-soft-gate-dc2108",
        "scheme": "shared_au_expert_soft_gate",
        "title": "Router soft-gate mass",
    },
    {
        "branch": "expert-grad-norm-dc2108",
        "scheme": "shared_au_expert_grad_norm",
        "title": "Expert gradient norm",
    },
    {
        "branch": "expert-update-similarity-dc2108",
        "scheme": "shared_au_expert_update_similarity",
        "title": "Expert update similarity",
    },
    {
        "branch": "expert-ema-frequency-dc2108",
        "scheme": "shared_au_expert_ema_freq",
        "title": "EMA activation frequency",
    },
    {
        "branch": "expert-scale-preserving-dc2108",
        "scheme": "shared_au_expert_scale_norm",
        "title": "Scale-preserving activation normalization",
    },
]


def run_git(args, capture=False):
    result = subprocess.run(
        ["git"] + args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot safely patch {label}: expected one anchor, found {count}."
        )
    return text.replace(old, new, 1)


def git_show_base(path):
    return run_git(["show", f"{BASE}:{path}"], capture=True)


def make_main(source):
    source = replace_once(
        source,
        "import copy\n",
        "import copy\nimport os\n",
        "main.py import",
    )

    assert_line = (
        '            "Expert mask dimension does not match model weights!"\n'
    )
    expert_setup = """
        # ========================================================
        # 五种专家参数聚合实验：共同基于 dc2108 基准版本
        # ========================================================

        expert_method_schemes = {
            'shared_au_expert_soft_gate',
            'shared_au_expert_grad_norm',
            'shared_au_expert_update_similarity',
            'shared_au_expert_ema_freq',
            'shared_au_expert_scale_norm'
        }

        use_expert_method = (
            weighting_method == 'adaptive'
            and aggregation_scheme in expert_method_schemes
        )

        if aggregation_scheme in expert_method_schemes and weighting_method != 'adaptive':
            raise ValueError(
                'The selected expert aggregation method requires '
                '-weighting adaptive.'
            )

        if use_expert_method:

            if not hasattr(model.model, 'num_experts'):
                raise ValueError(
                    'The selected method requires a MoE model.'
                )

            expert_num = model.model.num_experts
            expert_masks = []

            # 每个专家的参数 Mask，顺序与 model.get_weight() 完全一致
            for expert_idx in range(expert_num):

                mask_parts = []

                for key, value in model.model.state_dict().items():

                    is_this_expert = key.startswith(
                        f'experts.{expert_idx}.'
                    )

                    mask_parts.append(
                        torch.full(
                            (value.numel(),),
                            is_this_expert,
                            dtype=torch.bool,
                            device=value.device
                        )
                    )

                expert_masks.append(torch.cat(mask_parts))

            combined_expert_mask = torch.zeros_like(expert_mask)

            for mask in expert_masks:
                combined_expert_mask |= mask

            if not torch.equal(combined_expert_mask, expert_mask):
                raise ValueError(
                    'Per-expert masks do not match the global expert mask.'
                )

            method_log_filename = (
                'expertAggregationStats_' + results_file_prefix + '.csv'
            )

            if (
                not os.path.exists(method_log_filename)
                or os.path.getsize(method_log_filename) == 0
            ):

                columns = [
                    'sim_seed',
                    'num_iter',
                    'client_id',
                    'participated',
                    'fedau_weight',
                    'total_hard_activations'
                ]

                columns += [
                    f'expert_{e}_hard_count'
                    for e in range(expert_num)
                ]

                columns += [
                    f'expert_{e}_gate_mass'
                    for e in range(expert_num)
                ]

                columns += [
                    f'expert_{e}_gradient_norm'
                    for e in range(expert_num)
                ]

                columns += [
                    f'expert_{e}_method_weight'
                    for e in range(expert_num)
                ]

                with open(method_log_filename, 'a') as f:
                    f.write(','.join(columns) + '\\n')
"""
    source = replace_once(
        source, assert_line, assert_line + expert_setup, "expert method setup"
    )

    source = replace_once(
        source,
        """            if aggregation_scheme in (
                'shared_avg_expert_au',
                'shared_au_expert_avg'
            ) and (not has_expert or not has_shared):""",
        """            if aggregation_scheme in (
                'shared_avg_expert_au',
                'shared_au_expert_avg',
                'shared_au_expert_soft_gate',
                'shared_au_expert_grad_norm',
                'shared_au_expert_update_similarity',
                'shared_au_expert_ema_freq',
                'shared_au_expert_scale_norm'
            ) and (not has_expert or not has_shared):""",
        "aggregation scheme validation",
    )

    source = replace_once(
        source,
        """            elif aggregation_scheme == 'shared_au_expert_avg':
                fedavg_mask = expert_mask
                fedau_mask = shared_mask

            else:""",
        """            elif aggregation_scheme == 'shared_au_expert_avg':
                fedavg_mask = expert_mask
                fedau_mask = shared_mask

            elif aggregation_scheme in expert_method_schemes:
                # 五种新方法：Backbone + Router 使用 FedAU；
                # 专家参数在 FedAU 权重基础上再应用各自的方法权重。
                fedavg_mask = torch.zeros_like(expert_mask)
                fedau_mask = torch.ones_like(expert_mask)

            else:""",
        "aggregation masks",
    )

    source = replace_once(
        source,
        "        learning_rate_decay_level = 0\n",
        """        learning_rate_decay_level = 0

        # EMA 激活频率：每个客户端独立保存历史状态
        if use_expert_method:
            ema_activation_weights = torch.ones(
                n_nodes, expert_num, device=device
            )
            ema_activation_initialized = [False for _ in range(n_nodes)]
            expert_ema_beta = 0.9

            # 更新相似度：保存上一轮全局专家更新方向的 EMA
            similarity_reference = [None for _ in range(expert_num)]
            similarity_reference_beta = 0.9
""",
        "historical state initialization",
    )

    source = replace_once(
        source,
        """                # 本轮实际参与训练的客户端数量
                participating_count = 0
""",
        """                # 本轮实际参与训练的客户端数量
                participating_count = 0

                if use_expert_method:

                    shared_update_sum = torch.zeros_like(w_global)
                    expert_update_sum = torch.zeros_like(w_global)

                    expert_metric_sum = torch.zeros(
                        expert_num, dtype=torch.float32, device=device
                    )
                    expert_metric_count = torch.zeros(
                        expert_num, dtype=torch.float32, device=device
                    )

                    expert_weight_sum = torch.zeros(
                        expert_num, dtype=torch.float32, device=device
                    )
                    total_fedau_weight = 0.0

                    round_participated = [False for _ in range(n_nodes)]
                    round_hard_counts = torch.zeros(
                        n_nodes, expert_num, dtype=torch.long
                    )
                    round_gate_mass = torch.zeros(
                        n_nodes, expert_num, dtype=torch.float32
                    )
                    round_gradient_norms = torch.zeros(
                        n_nodes, expert_num, dtype=torch.float32
                    )
                    round_method_weights = torch.zeros(
                        n_nodes, expert_num, dtype=torch.float32
                    )
                    round_fedau_weights = torch.zeros(
                        n_nodes, dtype=torch.float32
                    )
""",
        "per-round accumulators",
    )

    source = replace_once(
        source,
        """                    model.model.train()

                    # ------------------------------------------------
                    # 本地 SGD 训练
""",
        """                    model.model.train()

                    if use_expert_method:

                        local_hard_counts = torch.zeros(
                            expert_num, dtype=torch.float32, device=device
                        )
                        local_gate_mass = torch.zeros(
                            expert_num, dtype=torch.float32, device=device
                        )
                        local_grad_norm_sum = torch.zeros(
                            expert_num, dtype=torch.float32, device=device
                        )

                    # ------------------------------------------------
                    # 本地 SGD 训练
""",
        "local statistic initialization",
    )

    source = replace_once(
        source,
        """                        output = model.model(images)

                        loss = model.loss_fn(output, labels)

                        loss.backward()

                        model.optimizer.step()
""",
        """                        output = model.model(images)

                        if use_expert_method:

                            local_hard_counts += (
                                model.model.last_expert_counts
                                .detach().to(device=device, dtype=torch.float32)
                            )

                            local_gate_mass += (
                                model.model.last_expert_gate_mass
                                .detach().to(device=device, dtype=torch.float32)
                            )

                        loss = model.loss_fn(output, labels)

                        loss.backward()

                        if aggregation_scheme == 'shared_au_expert_grad_norm':

                            # 在 optimizer.step() 之前统计每个专家的梯度范数
                            for expert_idx, expert in enumerate(model.model.experts):

                                squared_norm = torch.zeros(
                                    (), dtype=torch.float32, device=device
                                )

                                for parameter in expert.parameters():

                                    if parameter.grad is not None:
                                        squared_norm += (
                                            parameter.grad.detach()
                                            .float().pow(2).sum()
                                        )

                                local_grad_norm_sum[expert_idx] += torch.sqrt(
                                    squared_norm
                                )

                        model.optimizer.step()
""",
        "forward statistics and gradient norm",
    )

    source = replace_once(
        source,
        """                    # ------------------------------------------------
                    # 计算本地模型更新
                    # ------------------------------------------------

                    w_tmp = model.get_weight()
""",
        """                    # ------------------------------------------------
                    # 根据本地统计量计算专家方法权重
                    # ------------------------------------------------

                    if use_expert_method:

                        hard_total = local_hard_counts.sum()
                        gate_total = local_gate_mass.sum()

                        if hard_total.item() > 0:
                            hard_frequency = (
                                expert_num * local_hard_counts / hard_total
                            )
                        else:
                            hard_frequency = torch.ones(
                                expert_num, dtype=torch.float32, device=device
                            )

                        if gate_total.item() > 0:
                            soft_frequency = (
                                expert_num * local_gate_mass / gate_total
                            )
                        else:
                            soft_frequency = torch.ones(
                                expert_num, dtype=torch.float32, device=device
                            )

                        local_grad_norm = (
                            local_grad_norm_sum / max(iters_per_round, 1)
                        )

                        if aggregation_scheme == 'shared_au_expert_soft_gate':
                            local_method_weights = soft_frequency

                        elif aggregation_scheme == 'shared_au_expert_grad_norm':
                            local_method_weights = local_grad_norm

                        elif aggregation_scheme == 'shared_au_expert_update_similarity':
                            local_method_weights = torch.ones(
                                expert_num, dtype=torch.float32, device=device
                            )

                        elif aggregation_scheme == 'shared_au_expert_ema_freq':

                            if not ema_activation_initialized[n]:
                                ema_activation_weights[n] = hard_frequency.detach()
                                ema_activation_initialized[n] = True
                            else:
                                ema_activation_weights[n].mul_(
                                    expert_ema_beta
                                ).add_(
                                    hard_frequency.detach(),
                                    alpha=1.0 - expert_ema_beta
                                )

                            local_method_weights = ema_activation_weights[n].clone()

                        elif aggregation_scheme == 'shared_au_expert_scale_norm':
                            local_method_weights = hard_frequency

                        round_hard_counts[n] = (
                            local_hard_counts.detach().cpu().long()
                        )
                        round_gate_mass[n] = (
                            local_gate_mass.detach().cpu()
                        )
                        round_gradient_norms[n] = (
                            local_grad_norm.detach().cpu()
                        )

                    # ------------------------------------------------
                    # 计算本地模型更新
                    # ------------------------------------------------

                    w_tmp = model.get_weight()
""",
        "local expert factors",
    )

    source = replace_once(
        source,
        """                    elif weighting_method == 'mifa':

                        update_per_node[n] = w_tmp.to('cpu')

                    # ------------------------------------------------
                    # 记录聚合权重和参与次数
""",
        """                    elif weighting_method == 'mifa':

                        update_per_node[n] = w_tmp.to('cpu')

                    # 当前客户端 FedAU 系数；首次参与时按原始逻辑取 1
                    fedau_coeff = (
                        1.0 if agg_weight is None else float(agg_weight)
                    )

                    if use_expert_method:

                        if aggregation_scheme == 'shared_au_expert_update_similarity':

                            # 与上一轮全局专家更新方向比较。
                            # (cosine + 1) / 2 映射到 [0, 1]，避免直接丢弃
                            # Non-IID 环境中余弦略为负值的客户端更新。
                            similarity_weights = torch.ones(
                                expert_num, dtype=torch.float32, device=device
                            )

                            for expert_idx in range(expert_num):

                                client_update = w_tmp[
                                    expert_masks[expert_idx]
                                ].detach()

                                reference = similarity_reference[expert_idx]

                                if reference is not None:

                                    ref_norm = torch.linalg.vector_norm(reference)
                                    update_norm = torch.linalg.vector_norm(client_update)

                                    if (
                                        ref_norm.item() > 1e-12
                                        and update_norm.item() > 1e-12
                                    ):

                                        cosine = torch.dot(
                                            client_update, reference
                                        ) / (ref_norm * update_norm + 1e-12)

                                        similarity_weights[expert_idx] = (
                                            (cosine + 1.0) / 2.0
                                        ).clamp(0.0, 1.0)

                            local_method_weights = similarity_weights

                        round_method_weights[n] = (
                            local_method_weights.detach().cpu()
                        )
                        round_fedau_weights[n] = fedau_coeff
                        round_participated[n] = True

                    # ------------------------------------------------
                    # 记录聚合权重和参与次数
""",
        "FedAU coefficient and similarity",
    )

    source = replace_once(
        source,
        """                if use_scheme_aggregation:

                    if participated:

                        # FedAvg 参数：累积原始客户端更新，不乘 FedAU 权重
                        fedavg_update_sum[fedavg_mask] += (
                            w_tmp[fedavg_mask]
                        )

                        # FedAU 参数：累积上面已经按参与间隔加权的更新
                        fedau_update_sum[fedau_mask] += (
                            w_tmp[fedau_mask]
                        )

                        participating_count += 1

                elif weighting_method != 'mifa':
""",
        """                if use_scheme_aggregation:

                    if participated:

                        if use_expert_method:

                            # Backbone + Router：始终使用原始 FedAU
                            shared_update_sum[shared_mask] += (
                                w_tmp[shared_mask]
                            )

                            participating_count += 1

                            if aggregation_scheme == 'shared_au_expert_grad_norm':

                                # 先按本地专家梯度范数加权，轮末再按跨客户端均值归一化
                                for expert_idx in range(expert_num):

                                    mask = expert_masks[expert_idx]
                                    grad_value = local_grad_norm[expert_idx]

                                    expert_update_sum[mask] += (
                                        w_tmp[mask] * grad_value
                                    )

                                    expert_metric_sum[expert_idx] += grad_value
                                    expert_metric_count[expert_idx] += 1.0

                            elif aggregation_scheme == 'shared_au_expert_scale_norm':

                                # 先乘硬激活频率，轮末恢复每个专家的 FedAU 总权重尺度
                                total_fedau_weight += fedau_coeff

                                for expert_idx in range(expert_num):

                                    mask = expert_masks[expert_idx]
                                    factor = local_method_weights[expert_idx]

                                    expert_update_sum[mask] += (
                                        w_tmp[mask] * factor
                                    )

                                    expert_weight_sum[expert_idx] += (
                                        fedau_coeff * factor
                                    )

                            else:

                                # Router 软激活 / 更新相似度 / EMA 激活频率
                                for expert_idx in range(expert_num):

                                    mask = expert_masks[expert_idx]
                                    factor = local_method_weights[expert_idx]

                                    expert_update_sum[mask] += (
                                        w_tmp[mask] * factor
                                    )

                        else:

                            # 原有 FedAvg / FedAU 参数级聚合，保持不变
                            fedavg_update_sum[fedavg_mask] += (
                                w_tmp[fedavg_mask]
                            )

                            fedau_update_sum[fedau_mask] += (
                                w_tmp[fedau_mask]
                            )

                            participating_count += 1

                elif weighting_method != 'mifa':
""",
        "expert update accumulation",
    )

    source = replace_once(
        source,
        """            # ====================================================
            # 本轮客户端遍历结束
            # ====================================================

            if save_agg_weight:
""",
        """            # ====================================================
            # 本轮客户端遍历结束
            # ====================================================

            if use_expert_method:

                # 梯度范数方法的日志权重与聚合时使用同一归一化系数
                if aggregation_scheme == 'shared_au_expert_grad_norm':

                    for expert_idx in range(expert_num):

                        if expert_metric_count[expert_idx].item() > 0:

                            mean_grad = (
                                expert_metric_sum[expert_idx]
                                / expert_metric_count[expert_idx]
                            )

                            if mean_grad.item() > 1e-12:

                                round_method_weights[:, expert_idx] = (
                                    round_gradient_norms[:, expert_idx]
                                    / mean_grad.detach().cpu()
                                )

                with open(method_log_filename, 'a') as f:

                    for client_idx in range(n_nodes):

                        counts = round_hard_counts[client_idx]
                        gate_mass = round_gate_mass[client_idx]
                        grad_norms = round_gradient_norms[client_idx]
                        method_weights = round_method_weights[client_idx]

                        row = [
                            seed,
                            num_iter,
                            client_idx,
                            int(round_participated[client_idx]),
                            float(round_fedau_weights[client_idx]),
                            int(counts.sum().item())
                        ]

                        row += [int(x) for x in counts.tolist()]
                        row += [float(x) for x in gate_mass.tolist()]
                        row += [float(x) for x in grad_norms.tolist()]
                        row += [float(x) for x in method_weights.tolist()]

                        f.write(
                            ','.join(str(x) for x in row) + '\\n'
                        )

            if save_agg_weight:
""",
        "expert metrics logging",
    )

    source = replace_once(
        source,
        """            if use_scheme_aggregation:

                w_tmp_a = torch.zeros_like(w_global)

                # ------------------------------------------------
                # FedAvg 参数：
                # 对本轮实际参与训练的客户端更新求平均
                # ------------------------------------------------

                if participating_count > 0:

                    w_tmp_a[fedavg_mask] = (
                        fedavg_update_sum[fedavg_mask]
                        / participating_count
                    )

                # ------------------------------------------------
                # FedAU 参数：
                # 使用原始 Adaptive FedAU 的总客户端数归一化
                # 未参与客户端的更新视为零
                # ------------------------------------------------

                w_tmp_a[fedau_mask] = (
                    fedau_update_sum[fedau_mask] / n_nodes
                )

            elif weighting_method != 'mifa':
""",
        """            if use_scheme_aggregation:

                w_tmp_a = torch.zeros_like(w_global)

                if use_expert_method:

                    # Backbone + Router：原始 FedAU，除以系统总客户端数
                    w_tmp_a[shared_mask] = (
                        shared_update_sum[shared_mask] / n_nodes
                    )

                    for expert_idx in range(expert_num):

                        mask = expert_masks[expert_idx]

                        if aggregation_scheme == 'shared_au_expert_grad_norm':

                            # 专家梯度范数在参与客户端间归一化，平均因子为 1
                            if expert_metric_count[expert_idx].item() > 0:

                                mean_grad = (
                                    expert_metric_sum[expert_idx]
                                    / expert_metric_count[expert_idx]
                                )

                                if mean_grad.item() > 1e-12:

                                    w_tmp_a[mask] = (
                                        expert_update_sum[mask]
                                        / mean_grad
                                        / n_nodes
                                    )

                        elif aggregation_scheme == 'shared_au_expert_scale_norm':

                            # 将每个专家的加权系数总和恢复为本轮 FedAU 系数总和
                            denominator = expert_weight_sum[expert_idx]

                            if (
                                denominator.item() > 1e-12
                                and total_fedau_weight > 0.0
                            ):

                                scale = (
                                    total_fedau_weight / denominator
                                )

                                w_tmp_a[mask] = (
                                    expert_update_sum[mask]
                                    * scale
                                    / n_nodes
                                )

                        else:

                            # 软激活、相似度、EMA：保留 FedAU 的 1/N 尺度
                            w_tmp_a[mask] = (
                                expert_update_sum[mask] / n_nodes
                            )

                    # 更新相似度参考方向，供下一轮使用
                    if aggregation_scheme == 'shared_au_expert_update_similarity':

                        for expert_idx in range(expert_num):

                            current_reference = (
                                w_tmp_a[expert_masks[expert_idx]].detach()
                            )

                            if torch.linalg.vector_norm(
                                current_reference
                            ).item() > 1e-12:

                                if similarity_reference[expert_idx] is None:

                                    similarity_reference[expert_idx] = (
                                        current_reference.clone()
                                    )

                                else:

                                    similarity_reference[expert_idx].mul_(
                                        similarity_reference_beta
                                    ).add_(
                                        current_reference,
                                        alpha=1.0 - similarity_reference_beta
                                    )

                else:

                    # 原有 FedAvg / FedAU 参数级聚合，保持不变
                    if participating_count > 0:

                        w_tmp_a[fedavg_mask] = (
                            fedavg_update_sum[fedavg_mask]
                            / participating_count
                        )

                    w_tmp_a[fedau_mask] = (
                        fedau_update_sum[fedau_mask] / n_nodes
                    )

            elif weighting_method != 'mifa':
""",
        "expert final aggregation",
    )

    return source


def make_cnn(source):
    return replace_once(
        source,
        """        # Normalize selected routing weights
        topk_weights = F.softmax(topk_logits, dim=1)
""",
        """        # Normalize selected routing weights
        topk_weights = F.softmax(topk_logits, dim=1)

        # 路由统计仅用于服务器聚合权重，不参与计算图，不改变前向结果
        detached_indices = topk_indices.detach().reshape(-1)
        detached_weights = topk_weights.detach().reshape(-1)

        self.last_expert_counts = torch.bincount(
            detached_indices,
            minlength=self.num_experts
        )

        self.last_expert_gate_mass = torch.zeros(
            self.num_experts,
            dtype=topk_weights.dtype,
            device=features.device
        )

        self.last_expert_gate_mass.scatter_add_(
            0,
            detached_indices,
            detached_weights
        )
""",
        "cnn router statistics",
    )


def make_config(source, scheme):
    template = replace_once(
        source,
        """    default='shared_avg_expert_au',
    choices=[
        'fedavg_all',
        'fedau_all',
        'shared_avg_expert_au',
        'shared_au_expert_avg'
    ],""",
        """    default='__DEFAULT_SCHEME__',
    choices=[
        'fedavg_all',
        'fedau_all',
        'shared_avg_expert_au',
        'shared_au_expert_avg',
        'shared_au_expert_soft_gate',
        'shared_au_expert_grad_norm',
        'shared_au_expert_update_similarity',
        'shared_au_expert_ema_freq',
        'shared_au_expert_scale_norm'
    ],""",
        "config aggregation choices",
    )
    return template.replace("__DEFAULT_SCHEME__", scheme)


def main():
    root = Path(run_git(["rev-parse", "--show-toplevel"], capture=True))
    if Path.cwd().resolve() != root.resolve():
        print(f"Run this script from the repository root: {root}")
        return 2

    status_lines = run_git(["status", "--porcelain"], capture=True).splitlines()
    allowed_untracked = {
        "apply_expert_aggregation_variants.py",
        "README_expert_aggregation_variants.txt",
    }
    dirty_lines = []
    for line in status_lines:
        changed_path = line[3:].strip()
        if changed_path in allowed_untracked and line.startswith("??"):
            continue
        dirty_lines.append(line)

    if dirty_lines:
        print(
            "Working tree has other changes. Commit/stash them before running:\n"
            + "\n".join(dirty_lines)
        )
        return 2

    try:
        run_git(["cat-file", "-e", f"{BASE}^{{commit}}"])
    except subprocess.CalledProcessError:
        print(
            "Base commit is not present locally. Run:\n"
            f"  git fetch origin {BASE}\n"
            "Then run this script again."
        )
        return 2

    original_branch = run_git(["branch", "--show-current"], capture=True)
    original_head = run_git(["rev-parse", "HEAD"], capture=True)

    for variant in VARIANTS:
        existing = run_git(
            ["branch", "--list", variant["branch"]], capture=True
        )
        if existing:
            print(
                f"Branch already exists: {variant['branch']}. "
                "Rename/delete it before rerunning."
            )
            return 2

    base_main = git_show_base("main.py")
    base_config = git_show_base("config.py")
    base_cnn = git_show_base("model/cnn_cifar10.py")

    modified_main = make_main(base_main)
    modified_cnn = make_cnn(base_cnn)

    # Basic syntax check before creating any branches.
    compile(modified_main, "main.py", "exec")
    compile(modified_config := make_config(
        base_config, VARIANTS[0]["scheme"]
    ), "config.py", "exec")
    compile(modified_cnn, "model/cnn_cifar10.py", "exec")

    print("All generated Python files passed syntax compilation.")
    print(f"All five branches will start from {BASE}.")

    for variant in VARIANTS:
        branch = variant["branch"]
        print(f"\nCreating {branch} ...")

        run_git(["checkout", "-b", branch, BASE])

        branch_config = make_config(base_config, variant["scheme"])

        for path, content in [
            ("main.py", modified_main),
            ("config.py", branch_config),
            ("model/cnn_cifar10.py", modified_cnn),
        ]:
            with open(root / path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

        run_git(["add", "main.py", "config.py", "model/cnn_cifar10.py"])
        run_git([
            "commit",
            "-m",
            f"Add {variant['title']} expert aggregation from dc2108",
        ])

        head = run_git(["rev-parse", "HEAD"], capture=True)
        parent = run_git(["rev-parse", "HEAD^"], capture=True)

        if parent != BASE:
            raise RuntimeError(
                f"{branch} is not directly based on the requested commit."
            )

        print(f"Created {branch} at {head}; parent verified as {BASE}.")

    # Return to the branch/commit that was checked out before running.
    if original_branch:
        run_git(["checkout", original_branch])
    else:
        run_git(["checkout", "--detach", original_head])

    print("\nAll five local branches were created successfully.")
    print("To push them to GitHub, run:")
    for variant in VARIANTS:
        print(f"  git push -u origin {variant['branch']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
