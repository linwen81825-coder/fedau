
import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import data_participation_each_node
import numpy as np
import random
from model.model import Model
from util.util import DatasetSplit, WorkerSampler
import copy
import os


if device.type != 'cpu':
    torch.cuda.set_device(device)


if __name__ == "__main__":

    stat = CollectStatistics(
        results_eval_file_prefix=results_file_prefix
    )

    if save_agg_weight:

        agg_weight_filename = (
            'aggregationWeights_' + results_file_prefix + '.csv'
        )

        with open(agg_weight_filename, 'a') as f:
            f.write(
                'sim_seed,num_iter,aggregation_weight_each_node\n'
            )

        participation_count_filename = (
            'participationCount_' + results_file_prefix + '.csv'
        )

        with open(participation_count_filename, 'a') as f:
            f.write(
                'sim_seed,num_iter,participation_count_each_node\n'
            )

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        data_train, data_test = load_data(
            dataset,
            dataset_file_path,
            'cpu'
        )

        data_train_loader = DataLoader(
            data_train,
            batch_size=batch_size_eval,
            shuffle=True,
            num_workers=0
        )

        data_test_loader = DataLoader(
            data_test,
            batch_size=batch_size_eval,
            num_workers=0
        )

        dict_users, participation_prob_each_node = (
            data_participation_each_node(data_train, n_nodes)
        )

        step_size_local = step_size_local_config

        model = Model(
            seed,
            step_size_local,
            model_name=model_name,
            device=device,
            flatten_weight=True,
            pretrained_model_file=load_model_file
        )

        train_loader_list = []
        dataiter_list = []

        for n in range(n_nodes):

            train_loader_list.append(
                DataLoader(
                    DatasetSplit(data_train, dict_users[n]),
                    batch_size=batch_size_train,
                    shuffle=True
                )
            )

            dataiter_list.append(iter(train_loader_list[n]))

        def sample_minibatch(n):

            try:

                images, labels = next(dataiter_list[n])

                if len(images) < batch_size_train:

                    dataiter_list[n] = iter(train_loader_list[n])

                    images, labels = next(dataiter_list[n])

            except StopIteration:

                dataiter_list[n] = iter(train_loader_list[n])

                images, labels = next(dataiter_list[n])

            return images, labels

        def sample_full_batch(n):

            images = []
            labels = []

            for i in range(len(train_loader_list[n].dataset)):

                images.append(
                    train_loader_list[n].dataset[i][0]
                )

                l = train_loader_list[n].dataset[i][1]

                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)

                labels.append(l)

            return torch.stack(images), torch.stack(labels)

        # ========================================================
        # 初始化全局模型参数
        # ========================================================

        w_global = model.get_weight()

        # ========================================================
        # 区分专家参数和非专家参数
        # ========================================================

        expert_mask = model.get_expert_weight_mask()

        shared_mask = ~expert_mask

        assert expert_mask.numel() == w_global.numel(), \
            "Expert mask dimension does not match model weights!"

        # ========================================================
        # 新增：专家激活频率聚合方案
        # ========================================================

        track_expert_activation = (
            weighting_method == 'adaptive'
            and aggregation_scheme == 'shared_au_expert_freq_au'
        )

        if (
            aggregation_scheme == 'shared_au_expert_freq_au'
            and weighting_method != 'adaptive'
        ):
            raise ValueError(
                'shared_au_expert_freq_au requires -weighting adaptive.'
            )

        expert_masks = None
        expert_activation_log_filename = None

        if track_expert_activation:

            if not hasattr(model.model, 'num_experts'):
                raise ValueError(
                    'shared_au_expert_freq_au requires a MoE model '
                    'with num_experts.'
                )

            expert_masks = model.get_expert_weight_masks()

            if len(expert_masks) != model.model.num_experts:
                raise ValueError(
                    'The number of expert masks is incorrect.'
                )

            for mask in expert_masks:

                if mask.numel() != w_global.numel():
                    raise ValueError(
                        'Expert mask dimension does not match model weights!'
                    )

            # 专家激活次数日志
            expert_activation_log_filename = (
                'expertActivationCounts_'
                + results_file_prefix
                + '.csv'
            )

            if (
                not os.path.exists(expert_activation_log_filename)
                or os.path.getsize(expert_activation_log_filename) == 0
            ):

                log_columns = [
                    'sim_seed',
                    'num_iter',
                    'client_id',
                    'participated',
                    'total_activation_count'
                ]

                log_columns += [
                    f'expert_{e}_count'
                    for e in range(model.model.num_experts)
                ]

                log_columns += [
                    f'expert_{e}_frequency'
                    for e in range(model.model.num_experts)
                ]

                log_columns += [
                    f'expert_{e}_relative_weight'
                    for e in range(model.model.num_experts)
                ]

                with open(expert_activation_log_filename, 'a') as f:
                    f.write(','.join(log_columns) + '\n')

        # ========================================================
        # 根据 aggregation_scheme 选择 FedAvg / FedAU 参数掩码
        #
        # fedavg_all:
        #   Backbone + Router + Experts 全部使用 FedAvg
        #
        # fedau_all:
        #   Backbone + Router + Experts 全部使用原始 FedAU
        #
        # shared_avg_expert_au:
        #   非专家参数 FedAvg，专家参数 FedAU
        #
        # shared_au_expert_avg:
        #   非专家参数 FedAU，专家参数 FedAvg
        #
        # shared_au_expert_freq_au:
        #   非专家参数 FedAU
        #   专家参数 FedAU × 专家激活频率
        # ========================================================

        use_scheme_aggregation = (
            weighting_method == 'adaptive'
        )

        if use_scheme_aggregation:

            has_expert = torch.any(expert_mask).item()
            has_shared = torch.any(shared_mask).item()

            if aggregation_scheme in (
                'shared_avg_expert_au',
                'shared_au_expert_avg',
                'shared_au_expert_freq_au'
            ) and (not has_expert or not has_shared):

                raise ValueError(
                    'The selected aggregation scheme requires both '
                    'expert and non-expert parameters. Check '
                    'Model.get_expert_weight_mask() and the model names.'
                )

            if aggregation_scheme == 'fedavg_all':

                fedavg_mask = torch.ones_like(expert_mask)
                fedau_mask = torch.zeros_like(expert_mask)

            elif aggregation_scheme == 'fedau_all':

                fedavg_mask = torch.zeros_like(expert_mask)
                fedau_mask = torch.ones_like(expert_mask)

            elif aggregation_scheme == 'shared_avg_expert_au':

                fedavg_mask = shared_mask
                fedau_mask = expert_mask

            elif aggregation_scheme == 'shared_au_expert_avg':

                fedavg_mask = expert_mask
                fedau_mask = shared_mask

            elif aggregation_scheme == 'shared_au_expert_freq_au':

                # 所有参数先使用原始 FedAU
                # 专家参数随后额外乘以各自的激活频率
                fedavg_mask = torch.zeros_like(expert_mask)
                fedau_mask = torch.ones_like(expert_mask)

            else:

                raise ValueError(
                    'Unknown aggregation scheme: '
                    + str(aggregation_scheme)
                )

        else:

            fedavg_mask = None
            fedau_mask = None

        num_iter = 0

        last_output = 0
        last_save_latest = 0
        last_save_checkpoint = 0

        not_participate_count_at_node = []
        participate_intervals_at_node = []
        worker_samplers = []

        if save_agg_weight:

            agg_weight_each_node = []
            participation_count_each_node = []

        for n in range(n_nodes):

            not_participate_count_at_node.append(0)

            participate_intervals_at_node.append([])

            worker_samplers.append(
                WorkerSampler(
                    participation_method,
                    participation_prob_each_node[n]
                )
            )

            if save_agg_weight:

                agg_weight_each_node.append(None)
                participation_count_each_node.append(0)

        if weighting_method == 'fedvarp' or weighting_method == 'mifa':

            update_per_node = []

            for n in range(n_nodes):

                update_per_node.append(
                    torch.zeros(w_global.shape[0]).to('cpu')
                )

            update_per_node = torch.stack(update_per_node)

        if weighting_method == 'fedvarp':

            update_all_avg = torch.zeros(
                w_global.shape[0]
            ).to('cpu')

        learning_rate_decay_level = 0

        # ========================================================
        # 联邦训练循环
        # ========================================================

        while True:

            if save_agg_weight:

                with open(agg_weight_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))

                with open(participation_count_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))

            # ----------------------------------------------------
            # 学习率衰减：保持原始逻辑
            # ----------------------------------------------------

            if num_iter > lr_decay_start_iter:

                if (
                    int(
                        (num_iter - lr_decay_start_iter)
                        / lr_decay_interval
                    ) + 1 > learning_rate_decay_level
                    and learning_rate_decay_level < 10
                ):

                    learning_rate_decay_level += 1

                    step_size_local /= 2

                    model.update_learning_rate(step_size_local)

            # ----------------------------------------------------
            # 初始化本轮聚合变量
            # ----------------------------------------------------

            accumulated = 0

            if use_scheme_aggregation:

                fedavg_update_sum = torch.zeros_like(w_global)

                fedau_update_sum = torch.zeros_like(w_global)

                participating_count = 0

            # ====================================================
            # 新增：本轮客户端专家激活统计
            # ====================================================

            if track_expert_activation:

                expert_activation_counts_each_node = [
                    torch.zeros(
                        model.model.num_experts,
                        dtype=torch.long
                    )
                    for _ in range(n_nodes)
                ]

                expert_participated_each_node = [
                    False for _ in range(n_nodes)
                ]

            # ====================================================
            # 客户端本地训练
            # ====================================================

            for n in range(n_nodes):

                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():

                    participated = True

                    # 加载全局模型
                    model.assign_weight(w_global)

                    model.model.train()

                    # ------------------------------------------------
                    # 新增：初始化当前客户端专家激活计数
                    # ------------------------------------------------

                    if track_expert_activation:

                        local_expert_activation_counts = torch.zeros(
                            model.model.num_experts,
                            dtype=torch.long,
                            device=device
                        )

                    # ------------------------------------------------
                    # 本地 SGD 训练
                    # ------------------------------------------------

                    for i in range(0, iters_per_round):

                        images, labels = sample_minibatch(n)

                        images = images.to(device)
                        labels = labels.to(device)

                        if transform_train is not None:

                            images = transform_train(
                                images
                            ).contiguous()

                        model.optimizer.zero_grad()

                        output = model.model(images)

                        # --------------------------------------------
                        # 新增：累计本地训练中的专家激活次数
                        # --------------------------------------------

                        if track_expert_activation:

                            local_expert_activation_counts += (
                                model.model.last_expert_counts.detach()
                            )

                        loss = model.loss_fn(output, labels)

                        loss.backward()

                        model.optimizer.step()

                    # ------------------------------------------------
                    # 本地训练结束，保存专家激活统计
                    # ------------------------------------------------

                    if track_expert_activation:

                        expert_activation_counts_each_node[n] = (
                            local_expert_activation_counts.detach().cpu()
                        )

                        expert_participated_each_node[n] = True

                        total_activation_count = (
                            local_expert_activation_counts.sum()
                        )

                        if total_activation_count.item() > 0:

                            # 归一化激活频率
                            expert_frequency = (
                                model.model.num_experts
                                * local_expert_activation_counts.float()
                                / total_activation_count.float()
                            )

                        else:

                            expert_frequency = torch.ones(
                                model.model.num_experts,
                                dtype=torch.float32,
                                device=device
                            )

                    # ------------------------------------------------
                    # 计算本地模型更新
                    # ------------------------------------------------

                    w_tmp = model.get_weight()

                    w_tmp -= w_global

                    agg_weight = None

                    # ------------------------------------------------
                    # 原有聚合权重计算
                    # ------------------------------------------------

                    if weighting_method == 'known_prob':

                        w_tmp /= worker_sampler.participation_prob

                        agg_weight = (
                            1 / worker_sampler.participation_prob
                        )

                    elif weighting_method == 'adaptive':

                        if len(participate_intervals_at_node[n]) > 0:

                            agg_weight = np.mean(
                                participate_intervals_at_node[n]
                            )

                            if use_scheme_aggregation:

                                w_tmp[fedau_mask] *= agg_weight

                            else:

                                w_tmp *= agg_weight

                    elif weighting_method == 'fedvarp':

                        w_tmp_new = copy.deepcopy(w_tmp).to('cpu')

                        w_tmp -= update_per_node[n].to(device)

                        update_per_node[n] = w_tmp_new

                    elif weighting_method == 'mifa':

                        update_per_node[n] = w_tmp.to('cpu')

                    # ------------------------------------------------
                    # 新增：专家参数乘以各自的激活频率
                    # ------------------------------------------------

                    if track_expert_activation and participated:

                        for expert_idx in range(
                            model.model.num_experts
                        ):

                            w_tmp[expert_masks[expert_idx]] *= (
                                expert_frequency[expert_idx]
                            )

                    # ------------------------------------------------
                    # 记录聚合权重和参与次数
                    # ------------------------------------------------

                    if save_agg_weight:

                        agg_weight_each_node[n] = agg_weight

                        participation_count_each_node[n] += 1

                    # ------------------------------------------------
                    # 更新客户端参与间隔
                    # 保留原始 FedAU 机制
                    # ------------------------------------------------

                    participate_intervals_at_node[n].append(
                        not_participate_count_at_node[n] + 1
                    )

                    not_participate_count_at_node[n] = 0

                else:

                    participated = False

                    not_participate_count_at_node[n] += 1

                    if not_participate_count_at_node[n] >= k_value:

                        participate_intervals_at_node[n].append(
                            not_participate_count_at_node[n]
                        )

                        not_participate_count_at_node[n] = 0

                    w_tmp = 0.0

                    # 未参与客户端本轮激活次数记为零
                    if track_expert_activation:

                        expert_activation_counts_each_node[n] = (
                            torch.zeros(
                                model.model.num_experts,
                                dtype=torch.long
                            )
                        )

                # ------------------------------------------------
                # 保存聚合统计
                # ------------------------------------------------

                if save_agg_weight:

                    with open(agg_weight_filename, 'a') as f:

                        if agg_weight_each_node[n] is not None:

                            f.write(
                                ',' + "%.4f" % agg_weight_each_node[n]
                            )

                        else:

                            f.write(',nan')

                    with open(participation_count_filename, 'a') as f:

                        f.write(
                            ',' + str(participation_count_each_node[n])
                        )

                # =================================================
                # 两种参数分别累积
                # =================================================

                if use_scheme_aggregation:

                    if participated:

                        fedavg_update_sum[fedavg_mask] += (
                            w_tmp[fedavg_mask]
                        )

                        fedau_update_sum[fedau_mask] += (
                            w_tmp[fedau_mask]
                        )

                        participating_count += 1

                elif weighting_method != 'mifa':

                    if accumulated == 0:

                        w_accumulate = w_tmp

                    else:

                        w_accumulate += w_tmp

                    if (
                        weighting_method != 'average_participating'
                        and weighting_method != 'fedvarp'
                    ):

                        accumulated += 1

                    elif participated:

                        accumulated += 1

            # ====================================================
            # 本轮客户端遍历结束
            # ====================================================

            # ----------------------------------------------------
            # 新增：保存本轮所有客户端的专家激活日志
            # ----------------------------------------------------

            if track_expert_activation:

                with open(
                    expert_activation_log_filename,
                    'a'
                ) as f:

                    for n in range(n_nodes):

                        counts = (
                            expert_activation_counts_each_node[n]
                        )

                        total_count = int(counts.sum().item())

                        if total_count > 0:

                            frequencies = (
                                counts.float() / total_count
                            )

                            relative_weights = (
                                model.model.num_experts
                                * frequencies
                            )

                        else:

                            frequencies = torch.zeros(
                                model.model.num_experts,
                                dtype=torch.float32
                            )

                            relative_weights = torch.zeros_like(
                                frequencies
                            )

                        row = [
                            seed,
                            num_iter,
                            n,
                            int(expert_participated_each_node[n]),
                            total_count
                        ]

                        row += counts.tolist()

                        row += [
                            float(x)
                            for x in frequencies.tolist()
                        ]

                        row += [
                            float(x)
                            for x in relative_weights.tolist()
                        ]

                        f.write(
                            ','.join(
                                str(x) for x in row
                            ) + '\n'
                        )

            if save_agg_weight:

                with open(agg_weight_filename, 'a') as f:
                    f.write('\n')

                with open(participation_count_filename, 'a') as f:
                    f.write('\n')

            # ====================================================
            # 分别聚合非专家参数和专家参数
            # ====================================================

            if use_scheme_aggregation:

                w_tmp_a = torch.zeros_like(w_global)

                # ------------------------------------------------
                # FedAvg 参数
                # ------------------------------------------------

                if participating_count > 0:

                    w_tmp_a[fedavg_mask] = (
                        fedavg_update_sum[fedavg_mask]
                        / participating_count
                    )

                # ------------------------------------------------
                # FedAU 参数
                # ------------------------------------------------

                w_tmp_a[fedau_mask] = (
                    fedau_update_sum[fedau_mask] / n_nodes
                )

            elif weighting_method != 'mifa':

                if accumulated > 0:

                    w_tmp_a = torch.div(
                        w_accumulate,
                        torch.tensor(accumulated).to(device)
                    ).view(-1)

                    if weighting_method == 'fedvarp':

                        w_tmp_a += update_all_avg.to(device)

                        update_all_avg = torch.mean(
                            update_per_node,
                            0
                        )

                else:

                    w_tmp_a = torch.zeros(
                        w_global.shape[0]
                    ).to(device)

            else:

                w_tmp_a = torch.mean(
                    update_per_node,
                    0
                ).to(device)

            # ====================================================
            # 更新全局模型
            # ====================================================

            w_global += (
                torch.tensor(step_size_global).to(device)
                * w_tmp_a
            )

            num_iter = num_iter + iters_per_round

            # ====================================================
            # 保存 Checkpoint
            # ====================================================

            if (
                save_checkpoint
                and num_iter - last_save_checkpoint >= iters_checkpoint
            ):

                torch.save(
                    model.model.state_dict(),
                    save_model_file
                    + '-checkpoint-sim-'
                    + str(seed)
                    + '-iter-'
                    + str(num_iter)
                )

                last_save_checkpoint = num_iter

            # ====================================================
            # 模型评估
            # ====================================================

            if num_iter - last_output >= min_iters_per_eval:

                stat.collect_stat_eval(
                    seed,
                    num_iter,
                    model,
                    data_train_loader,
                    data_test_loader,
                    w_global
                )

                last_output = num_iter

            # ====================================================
            # 结束条件
            # ====================================================

            if num_iter >= max_iter:

                break

        # ========================================================
        # 保存最终模型
        # ========================================================

        if save_final_model:

            torch.save(
                model.model.state_dict(),
                'final_model_sim'
                + str(seed)
                + '_iter'
                + str(num_iter)
                + '_'
                + save_model_file
            )

        del model
        del w_global

        if weighting_method != 'mifa' and not use_scheme_aggregation:

            del w_accumulate

        torch.cuda.empty_cache()
