# ============================================================
# 控制台仅显示每次评估的核心指标和历史最佳测试准确率；其余 stdout/stderr 全部保存到日志
# 日志路径：logs/<-out参数>_full.log
# ============================================================
import os
import sys
from datetime import datetime


def _init_full_log():
    output_name = "training"
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "-out" and index + 1 < len(args):
            output_name = args[index + 1]
        elif arg.startswith("-out="):
            output_name = arg.split("=", 1)[1]

    safe_name = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in os.path.basename(output_name)
    ).strip("._") or "training"

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, safe_name + "_full.log")

    # 保存原始终端输出句柄，供指标行使用。
    console_stdout = os.fdopen(os.dup(1), "w", buffering=1)
    console_stderr = os.fdopen(os.dup(2), "w", buffering=1)

    # 追加写入，避免同名实验覆盖之前的完整日志。
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = log_file
    sys.stderr = log_file

    # 将 Python 原始标准流也指向完整日志，避免调试输出泄漏到终端。
    sys.__stdout__ = log_file
    sys.__stderr__ = log_file

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Run started: " + datetime.now().astimezone().isoformat() + "\n")
    log_file.write("Full log file: " + os.path.abspath(log_path) + "\n")
    log_file.flush()

    return console_stdout, console_stderr, log_file, log_path


_CONSOLE_STDOUT, _CONSOLE_STDERR, _FULL_LOG_FILE, FULL_LOG_PATH = _init_full_log()


def _print_round_metrics(round_number, train_metrics, test_metrics, best_test_accuracy):
    train_loss, train_accuracy = train_metrics
    test_loss, test_accuracy = test_metrics

    line = (
        f"Round {round_number:>5d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy * 100:.2f}% | "
        f"Best Acc: {best_test_accuracy * 100:.2f}%"
    )

    # 指标行同时保存在完整日志中，并显示在控制台。
    _FULL_LOG_FILE.write(line + "\n")
    _FULL_LOG_FILE.flush()
    _CONSOLE_STDOUT.write(line + "\n")
    _CONSOLE_STDOUT.flush()


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


def _evaluate_loader(model, data_loader, global_weight, eval_device, transform=None):
    """备用评估函数：按样本数正确计算平均交叉熵和准确率。"""
    model.assign_weight(global_weight)
    model.model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)

            if transform is not None:
                images = transform(images).contiguous()

            logits = model.model(images)
            total_loss += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        return float("nan"), float("nan")

    return total_loss / total_samples, total_correct / total_samples


def _collect_eval_metrics(stat, seed, num_iter, model, train_loader,
                          test_loader, global_weight, eval_device):
    """保留原统计/CSV流程，同时捕获 train/test loss 和 accuracy。"""
    captured = {}
    original_accuracy = model.accuracy
    original_loss_fn = model.loss_fn

    def _tracked_accuracy(data_loader, weight, device_arg, transform=None):
        result = original_accuracy(data_loader, weight, device_arg, transform)
        if data_loader is train_loader:
            captured["train"] = result
        elif data_loader is test_loader:
            captured["test"] = result
        return result

    # 评估时临时使用 sum reduction，使累计损失除以样本数后成为
    # 正确的平均 CE loss；评估结束后立即恢复训练用 loss_fn。
    model.accuracy = _tracked_accuracy
    model.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(eval_device)

    try:
        stat.collect_stat_eval(
            seed, num_iter, model, train_loader, test_loader, global_weight
        )
    finally:
        model.accuracy = original_accuracy
        model.loss_fn = original_loss_fn

    # 兼容未通过 model.accuracy 返回某项指标的 CollectStatistics 版本。
    if "train" not in captured:
        captured["train"] = _evaluate_loader(
            model, train_loader, global_weight, eval_device,
            globals().get("transform_train", None)
        )
    if "test" not in captured:
        captured["test"] = _evaluate_loader(
            model, test_loader, global_weight, eval_device,
            globals().get("transform_test", None)
        )

    return captured["train"], captured["test"]


if device.type != 'cpu':
    torch.cuda.set_device(device)


if __name__ == "__main__":
    stat = CollectStatistics(results_eval_file_prefix=results_file_prefix)

    if save_agg_weight:
        agg_weight_filename = 'aggregationWeights_' + results_file_prefix + '.csv'
        with open(agg_weight_filename, 'a') as f:
            f.write('sim_seed,num_iter,aggregation_weight_each_node\n')
        participation_count_filename = 'participationCount_' + results_file_prefix + '.csv'
        with open(participation_count_filename, 'a') as f:
            f.write('sim_seed,num_iter,participation_count_each_node\n')

    for seed in simulations:

        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=batch_size_eval, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
        dict_users, participation_prob_each_node = data_participation_each_node(data_train, n_nodes)

        step_size_local = step_size_local_config
        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)

        train_loader_list = []
        dataiter_list = []
        for n in range(n_nodes):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
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
                images.append(train_loader_list[n].dataset[i][0])

                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)

        w_global = model.get_weight()   # Get initial weight
        # =========================================
        # MoE: 判断是否存在专家参数
        # =========================================

        has_experts = bool(
            model.expert_weight_mask.any().item()
        )

        if has_experts and weighting_method not in ('adaptive', 'average_participating'):
            raise ValueError(
                "MoE模式下 -weighting 仅支持 adaptive 或 average_participating"
            )

        if has_experts:

            non_expert_global, expert_global = model.split_weight(
                w_global
            )

            print("MoE双路聚合已启用")
            print("专家参数聚合方式:", weighting_method)
            print("非专家参数聚合方式: average_participating")
            print(
                "非专家参数数量:",
                non_expert_global.numel()
            )
            print(
                "专家参数数量:",
                expert_global.numel()
            )

        num_iter = 0
        best_test_accuracy = 0.0  # 当前 seed 历史最高 Test Acc
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
            worker_samplers.append(WorkerSampler(participation_method, participation_prob_each_node[n]))
            if save_agg_weight:
                agg_weight_each_node.append(None)
                participation_count_each_node.append(0)

        if weighting_method == 'fedvarp' or weighting_method == 'mifa':
            update_per_node = []
            for n in range(n_nodes):
                update_per_node.append(torch.zeros(w_global.shape[0]).to('cpu'))
            update_per_node = torch.stack(update_per_node)
        if weighting_method == 'fedvarp':
            update_all_avg = torch.zeros(w_global.shape[0]).to('cpu')

        learning_rate_decay_level = 0

        while True:
            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))
                with open(participation_count_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))

            if num_iter > lr_decay_start_iter:
                if int((num_iter - lr_decay_start_iter) / lr_decay_interval) + 1 > learning_rate_decay_level and \
                        learning_rate_decay_level < 10:
                    learning_rate_decay_level += 1
                    step_size_local /= 2
                    model.update_learning_rate(step_size_local)

            print('seed', seed, ' - iteration', num_iter, ' - local step size', step_size_local)
            accumulated = 0
            if has_experts:

                # 非专家参数：参与客户端等权平均
                non_expert_accumulate = torch.zeros_like(
                    non_expert_global
                )

                # 专家参数：FedAU聚合
                expert_accumulate = torch.zeros_like(
                    expert_global
                )

                # 本轮实际参与训练的客户端数量
                num_participating = 0
            for n in range(n_nodes):
                worker_sampler = worker_samplers[n]

                # 每个客户端每轮重新初始化，避免未参与客户端沿用上一轮的权重记录。
                agg_weight = None

                if worker_sampler.sample():
                    participated = True

                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, iters_per_round):
                        images, labels = sample_minibatch(n)

                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous()  # contiguous() needed due to the use of ColorJitter in CIFAR transforms

                        model.optimizer.zero_grad()
                        output = model.model(images)

                        # 分类损失
                        loss = model.loss_fn(output, labels)

                        # MoE Router负载均衡辅助损失
                        aux_loss = getattr(
                            model.model,
                            'aux_loss',
                            None
                        )

                        if aux_loss is not None:
                            loss = loss + moe_aux_loss_weight * aux_loss

                        loss.backward()
                        model.optimizer.step()

                    w_tmp = model.get_weight()
                    w_tmp -= w_global

                    if has_experts:

                        # =====================================
                        # 1. 拆分客户端模型更新
                        # =====================================

                        non_expert_delta, expert_delta = model.split_weight(
                            w_tmp
                        )

                        # =====================================
                        # 2. 专家参数按命令行指定方式处理
                        # adaptive: 原始FedAU自适应权重
                        # average_participating: 不加权，参与客户端等权平均
                        # =====================================

                        if weighting_method == 'adaptive':

                            if len(participate_intervals_at_node[n]) > 0:
                                agg_weight = np.mean(
                                    participate_intervals_at_node[n]
                                )
                                expert_delta *= agg_weight

                        elif weighting_method == 'average_participating':
                            # 保持原始客户端更新，不乘FedAU自适应权重
                            agg_weight = None

                        # =====================================
                        # 3. 分别累积两组更新
                        # =====================================

                        # 非专家参数不乘FedAU权重
                        non_expert_accumulate += non_expert_delta

                        # 累积当前实验设置下的专家参数更新
                        expert_accumulate += expert_delta

                        num_participating += 1

                    else:

                        # =====================================
                        # 非MoE模型：保持原来的聚合代码
                        # =====================================

                        if weighting_method == 'known_prob':

                            w_tmp /= worker_sampler.participation_prob
                            agg_weight = 1 / worker_sampler.participation_prob

                        elif weighting_method == 'adaptive':

                            if len(participate_intervals_at_node[n]) > 0:

                                agg_weight = np.mean(
                                    participate_intervals_at_node[n]
                                )

                                w_tmp *= agg_weight

                        elif weighting_method == 'fedvarp':

                            w_tmp_new = copy.deepcopy(w_tmp).to('cpu')

                            w_tmp -= update_per_node[n].to(device)

                            update_per_node[n] = w_tmp_new

                        elif weighting_method == 'mifa':

                            update_per_node[n] = w_tmp.to('cpu')

                    if save_agg_weight:
                        participation_count_each_node[n] += 1

                    participate_intervals_at_node[n].append(not_participate_count_at_node[n] + 1)  # interval used in the next round
                    not_participate_count_at_node[n] = 0

                else:
                    participated = False
                    not_participate_count_at_node[n] += 1

                    if not_participate_count_at_node[n] >= k_value:
                        participate_intervals_at_node[n].append(not_participate_count_at_node[n])
                        not_participate_count_at_node[n] = 0

                    w_tmp = 0.0

                if save_agg_weight:
                    # 未参与的客户端记录 nan，不沿用上一轮的自适应权重。
                    agg_weight_each_node[n] = agg_weight
                    with open(agg_weight_filename, 'a') as f:
                        if agg_weight_each_node[n] is not None:
                            f.write(',' + "%.4f" % agg_weight_each_node[n])
                        else:
                            f.write(',nan')
                    with open(participation_count_filename, 'a') as f:
                        f.write(',' + str(participation_count_each_node[n]))

                if not has_experts:

                    if weighting_method != 'mifa':

                        if accumulated == 0:
                            w_accumulate = w_tmp
                        else:
                            w_accumulate += w_tmp

                        if weighting_method != 'average_participating' and weighting_method != 'fedvarp':
                            accumulated += 1
                        elif participated:
                            accumulated += 1

            if has_experts:

                # =====================================
                # 1. 非专家参数：average_participating
                # =====================================

                if num_participating > 0:

                    non_expert_update = (
                        non_expert_accumulate / num_participating
                    )

                else:

                    non_expert_update = torch.zeros_like(
                        non_expert_accumulate
                    )

                # =====================================
                # 2. 专家参数：根据实验设置选择聚合方式
                # =====================================

                if weighting_method == 'adaptive':
                    # 原始FedAU：分母为全部客户端数量
                    expert_update = expert_accumulate / n_nodes

                elif weighting_method == 'average_participating':
                    # 客户端等权平均：分母为本轮实际参与客户端数量
                    if num_participating > 0:
                        expert_update = expert_accumulate / num_participating
                    else:
                        expert_update = torch.zeros_like(expert_accumulate)

                # =====================================
                # 3. 重新组合两组参数
                # =====================================

                w_tmp_a = model.combine_weight(
                    non_expert_update,
                    expert_update
                )

            else:

                # =====================================
                # 非MoE模型：保留原来的聚合方式
                # =====================================

                if weighting_method != 'mifa':

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

            w_global += torch.tensor(step_size_global).to(device) * w_tmp_a

            num_iter = num_iter + iters_per_round

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                model.assign_weight(w_global)
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                train_metrics, test_metrics = _collect_eval_metrics(
                    stat=stat,
                    seed=seed,
                    num_iter=num_iter,
                    model=model,
                    train_loader=data_train_loader,
                    test_loader=data_test_loader,
                    global_weight=w_global,
                    eval_device=device,
                )
                round_number = (num_iter + iters_per_round - 1) // iters_per_round
                current_test_accuracy = test_metrics[1]
                if np.isfinite(current_test_accuracy):
                    best_test_accuracy = max(best_test_accuracy, current_test_accuracy)
                _print_round_metrics(
                    round_number, train_metrics, test_metrics, best_test_accuracy
                )
                last_output = num_iter

            if num_iter >= max_iter:
                break

        if save_final_model:
            # 保存聚合后的全局模型，而不是最后一个客户端的本地模型
            model.assign_weight(w_global)
            torch.save(model.model.state_dict(), 'final_model_sim' + str(seed) + '_iter' + str(num_iter) + '_' + save_model_file)

        del model
        del w_global
        # MoE 双路聚合不使用原始整模型累积变量 w_accumulate
        if weighting_method != 'mifa' and not has_experts:
            del w_accumulate

        torch.cuda.empty_cache()
