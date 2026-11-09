
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
        # 新增：区分专家参数和非专家参数
        # ========================================================

        expert_mask = model.get_expert_weight_mask()

        shared_mask = ~expert_mask

        assert expert_mask.numel() == w_global.numel(), \
            "Expert mask dimension does not match model weights!"

        # ========================================================
        # 根据 aggregation_scheme 选择 FedAvg / FedAU 参数掩码
        #
        # fedavg_all:
        #   Backbone + Router + Experts 全部使用 FedAvg
        # fedau_all:
        #   Backbone + Router + Experts 全部使用原始 FedAU
        # shared_avg_expert_au:
        #   非专家参数 FedAvg，专家参数 FedAU
        # shared_au_expert_avg:
        #   非专家参数 FedAU，专家参数 FedAvg
        # ========================================================

        use_scheme_aggregation = (weighting_method == 'adaptive')

        if use_scheme_aggregation:

            has_expert = torch.any(expert_mask).item()
            has_shared = torch.any(shared_mask).item()

            if aggregation_scheme in (
                'shared_avg_expert_au',
                'shared_au_expert_avg'
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

                # FedAvg 参数的未加权更新累积
                fedavg_update_sum = torch.zeros_like(w_global)

                # FedAU 参数的参与间隔加权更新累积
                fedau_update_sum = torch.zeros_like(w_global)

                # 本轮实际参与训练的客户端数量
                participating_count = 0

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

                        loss = model.loss_fn(output, labels)

                        loss.backward()

                        model.optimizer.step()

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

                        agg_weight = 1 / worker_sampler.participation_prob

                    elif weighting_method == 'adaptive':

                        if len(participate_intervals_at_node[n]) > 0:

                            agg_weight = np.mean(
                                participate_intervals_at_node[n]
                            )

                            if use_scheme_aggregation:

                                # 只对本实验方案中指定为 FedAU 的参数加权
                                w_tmp[fedau_mask] *= agg_weight

                            else:

                                # 兼容未启用参数级聚合方案的原始逻辑
                                w_tmp *= agg_weight

                    elif weighting_method == 'fedvarp':

                        w_tmp_new = copy.deepcopy(w_tmp).to('cpu')

                        w_tmp -= update_per_node[n].to(device)

                        update_per_node[n] = w_tmp_new

                    elif weighting_method == 'mifa':

                        update_per_node[n] = w_tmp.to('cpu')

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
                # 核心修改：两种参数分别累积
                # =================================================

                if use_scheme_aggregation:

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

                    # 其他聚合算法保持原来的累积方式

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

            if save_agg_weight:

                with open(agg_weight_filename, 'a') as f:
                    f.write('\n')

                with open(participation_count_filename, 'a') as f:
                    f.write('\n')

            # ====================================================
            # 核心修改：分别聚合非专家参数和专家参数
            # ====================================================

            if use_scheme_aggregation:

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

                # ------------------------------------------------
                # 其他聚合方式保持原始逻辑
                # ------------------------------------------------

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
