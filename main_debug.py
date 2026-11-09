import torch
import torch.nn as nn
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
import math


# ============================================================
# DEBUG HELPERS
# 仅用于定位本地训练数值爆炸，不修改梯度、不裁剪、不改变聚合算法
# ============================================================

DEBUG_GRAD_THRESHOLD = 1.0e3
DEBUG_GRAD_CLIP_MAX_NORM = 1.0  # 全模型全局梯度范数裁剪阈值；不改变学习率
DEBUG_PARAM_THRESHOLD = 1.0e3
DEBUG_LOGIT_THRESHOLD = 1.0e3
DEBUG_ACTIVATION_THRESHOLD = 1.0e3
DEBUG_TOPK = 5

_DEBUG_CONTEXT = {
    "seed": None,
    "round_iter": None,
    "client": None,
    "local_step": None,
    "activations": [],
}


def _debug_context_text():
    return (
        "seed={seed}, round_iter={round_iter}, client={client}, "
        "local_step={local_step}"
    ).format(**_DEBUG_CONTEXT)


def _debug_check_finite(name, tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return

    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        raise FloatingPointError(
            "[NON-FINITE] {name}; {context}; nan={nan}, inf={inf}, shape={shape}".format(
                name=name,
                context=_debug_context_text(),
                nan=nan_count,
                inf=inf_count,
                shape=tuple(tensor.shape),
            )
        )


def _debug_max_abs(tensor):
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().item())


def _debug_named_tensor_stats(named_tensors, label, threshold=None):
    records = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        _debug_check_finite(label + ":" + name, tensor)
        value = _debug_max_abs(tensor)
        records.append((value, name))

    records.sort(reverse=True)
    if records:
        print(
            "[{label}] {context}; top={records}".format(
                label=label,
                context=_debug_context_text(),
                records=[
                    (name, value)
                    for value, name in records[:DEBUG_TOPK]
                ],
            ),
            flush=True,
        )

    max_value, max_name = records[0] if records else (0.0, "none")
    if threshold is not None and max_value > threshold:
        raise FloatingPointError(
            "[LARGE-{label}] {context}; max_abs={value:.6e} ({name}), "
            "threshold={threshold:.6e}".format(
                label=label,
                context=_debug_context_text(),
                value=max_value,
                name=max_name,
                threshold=threshold,
            )
        )
    return max_value, max_name


def _debug_activation_hook(name):
    def hook(module, inputs, output):
        if not torch.is_tensor(output):
            return

        _debug_check_finite("activation:" + name, output)
        max_abs = _debug_max_abs(output)
        _DEBUG_CONTEXT["activations"].append(
            (max_abs, name, tuple(output.shape))
        )

        if max_abs > DEBUG_ACTIVATION_THRESHOLD:
            raise FloatingPointError(
                "[LARGE-ACTIVATION] layer={name}; {context}; "
                "max_abs={value:.6e}, shape={shape}".format(
                    name=name,
                    context=_debug_context_text(),
                    value=max_abs,
                    shape=tuple(output.shape),
                )
            )
    return hook


def _debug_register_activation_hooks(model):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(
                module.register_forward_hook(_debug_activation_hook(name))
            )
    return handles


def _debug_print_top_activations():
    records = sorted(
        _DEBUG_CONTEXT["activations"],
        key=lambda item: item[0],
        reverse=True,
    )
    print(
        "[ACTIVATIONS] {context}; top={records}".format(
            context=_debug_context_text(),
            records=[
                (name, value, shape)
                for value, name, shape in records[:DEBUG_TOPK]
            ],
        ),
        flush=True,
    )


def _debug_client_label_histogram(data_train, client_indices):
    targets = data_train.targets
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.asarray(targets)

    client_labels = targets[np.asarray(client_indices, dtype=np.int64)]
    histogram = np.bincount(client_labels.astype(np.int64), minlength=10)
    total = int(histogram.sum())
    dominant_ratio = float(histogram.max() / total) if total > 0 else 0.0

    print(
        "[CLIENT-DATA] client={client}, samples={total}, "
        "label_hist={hist}, dominant_ratio={ratio:.4f}".format(
            client=_DEBUG_CONTEXT["client"],
            total=total,
            hist=histogram.tolist(),
            ratio=dominant_ratio,
        ),
        flush=True,
    )


def _debug_check_scalar(name, value):
    if torch.is_tensor(value):
        _debug_check_finite(name, value)
        value = float(value.detach().item())
    else:
        value = float(value)

    if not np.isfinite(value):
        raise FloatingPointError(
            "[NON-FINITE] {name}={value}; {context}".format(
                name=name,
                value=value,
                context=_debug_context_text(),
            )
        )
    return value


def _debug_check_parameter_magnitude(model, threshold=DEBUG_PARAM_THRESHOLD):
    return _debug_named_tensor_stats(
        model.named_parameters(),
        "PARAMETERS",
        threshold=threshold,
    )


def _debug_check_gradient_magnitude(model, threshold=DEBUG_GRAD_THRESHOLD):
    return _debug_named_tensor_stats(
        (
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ),
        "GRADIENTS",
        threshold=threshold,
    )


def _debug_global_gradient_l2(model):
    """使用 float64 累加全模型梯度 L2 Norm，避免统计时低精度溢出。"""
    total_sq = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()
        _debug_check_finite("gradient_after_clip", grad)

        grad64 = grad.to(dtype=torch.float64)
        total_sq += float(torch.sum(grad64 * grad64).item())

    return math.sqrt(total_sq)


def _debug_log_gradient_norms(model):
    """
    记录全模型梯度 L2 Norm，以及每个参数层的梯度 L2 Norm 和最大绝对值。
    只读取梯度，不修改梯度、不裁剪梯度，也不改变优化器更新。
    """
    total_sq = 0.0
    layer_sq_sums = {}
    layer_max_abs = {}
    gradient_tensor_count = 0
    nonfinite_parameters = []

    for name, parameter in model.named_parameters():
        grad = parameter.grad

        if grad is None:
            continue

        gradient_tensor_count += 1
        grad = grad.detach()

        if not bool(torch.isfinite(grad).all().item()):
            nonfinite_parameters.append(name)
            continue

        # 使用 float64 累加平方和，避免统计梯度范数时发生低精度溢出。
        grad64 = grad.to(dtype=torch.float64)
        norm_sq = float(torch.sum(grad64 * grad64).item())

        # 将同一模块的 weight / bias 合并为一层统计。
        layer_name = name.rsplit(".", 1)[0] if "." in name else name
        layer_sq_sums[layer_name] = (
            layer_sq_sums.get(layer_name, 0.0) + norm_sq
        )

        max_abs = float(grad.abs().max().item())
        layer_max_abs[layer_name] = max(
            layer_max_abs.get(layer_name, 0.0),
            max_abs,
        )

        total_sq += norm_sq

    if nonfinite_parameters:
        raise FloatingPointError(
            "[GRAD-NONFINITE] {context}; parameters={parameters}".format(
                context=_debug_context_text(),
                parameters=nonfinite_parameters,
            )
        )

    global_l2 = math.sqrt(total_sq)
    global_max_abs = max(layer_max_abs.values()) if layer_max_abs else 0.0

    print(
        "[GRAD-NORM] {context}; global_l2={global_l2:.8e}, "
        "global_max_abs={global_max_abs:.8e}, grad_tensors={count}".format(
            context=_debug_context_text(),
            global_l2=global_l2,
            global_max_abs=global_max_abs,
            count=gradient_tensor_count,
        ),
        flush=True,
    )

    # 按每层 L2 Norm 从大到小输出，便于定位梯度主要集中在哪一层。
    sorted_layers = sorted(
        layer_sq_sums.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for layer_name, norm_sq in sorted_layers:
        print(
            "[GRAD-LAYER] {context}; layer={layer}; "
            "l2={l2:.8e}, max_abs={max_abs:.8e}".format(
                context=_debug_context_text(),
                layer=layer_name,
                l2=math.sqrt(norm_sq),
                max_abs=layer_max_abs[layer_name],
            ),
            flush=True,
        )




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

        # 记录 Conv/Linear 层的前向输出；不改变模型计算
        debug_hook_handles = _debug_register_activation_hooks(model.model)

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

                if worker_sampler.sample():
                    participated = True

                    _DEBUG_CONTEXT.update({
                        "seed": seed,
                        "round_iter": num_iter,
                        "client": n,
                        "local_step": "before_local_training",
                        "activations": [],
                    })

                    _debug_client_label_histogram(data_train, dict_users[n])

                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, iters_per_round):
                        _DEBUG_CONTEXT.update({
                            "seed": seed,
                            "round_iter": num_iter,
                            "client": n,
                            "local_step": i,
                            "activations": [],
                        })

                        images, labels = sample_minibatch(n)
                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous()

                        _debug_check_finite("input_images", images)
                        _debug_check_finite("input_labels", labels)

                        print(
                            "[BATCH] {context}; image_min={image_min:.6e}, "
                            "image_max={image_max:.6e}, image_abs_max={image_abs:.6e}, "
                            "labels={labels}".format(
                                context=_debug_context_text(),
                                image_min=float(images.min().item()),
                                image_max=float(images.max().item()),
                                image_abs=_debug_max_abs(images),
                                labels=labels.detach().cpu().tolist(),
                            ),
                            flush=True,
                        )

                        _debug_check_parameter_magnitude(
                            model.model,
                            threshold=DEBUG_PARAM_THRESHOLD,
                        )

                        model.optimizer.zero_grad()
                        output = model.model(images)

                        _debug_check_finite("local_logits", output)
                        logits_max = _debug_max_abs(output)
                        _debug_print_top_activations()

                        ce_loss = model.loss_fn(output, labels)
                        _debug_check_scalar("ce_loss", ce_loss)

                        # MoE Router负载均衡辅助损失
                        aux_loss = getattr(model.model, "aux_loss", None)
                        if aux_loss is not None:
                            aux_value = _debug_check_scalar("router_aux_loss", aux_loss)
                            loss = ce_loss + moe_aux_loss_weight * aux_loss
                        else:
                            aux_value = 0.0
                            loss = ce_loss

                        total_loss_value = _debug_check_scalar("total_loss", loss)

                        expert_indices = getattr(
                            model.model,
                            "last_expert_indices",
                            None,
                        )
                        if expert_indices is not None:
                            # Top-1时索引形状为[batch_size]；
                            # Top-2时索引形状为[batch_size, 2]。
                            # 展平后统计每个专家收到的路由分配次数。
                            expert_counts = torch.bincount(
                                expert_indices.reshape(-1),
                                minlength=model.model.num_experts,
                            )
                            expert_counts = expert_counts.detach().cpu().tolist()
                        else:
                            expert_counts = None

                        print(
                            "[FORWARD] {context}; CE={ce:.6e}, AUX={aux:.6e}, "
                            "TOTAL={total:.6e}, logits_abs_max={logits:.6e}, "
                            "expert_counts={counts}".format(
                                context=_debug_context_text(),
                                ce=float(ce_loss.detach().item()),
                                aux=aux_value,
                                total=total_loss_value,
                                logits=logits_max,
                                counts=expert_counts,
                            ),
                            flush=True,
                        )

                        if logits_max > DEBUG_LOGIT_THRESHOLD:
                            raise FloatingPointError(
                                "[LARGE-LOGITS] {context}; max_abs={value:.6e}".format(
                                    context=_debug_context_text(),
                                    value=logits_max,
                                )
                            )

                        loss.backward()

                        # 记录裁剪前的原始梯度；该函数同时检查 NaN/Inf。
                        # 原始梯度的层级 L2 和 max_abs 会保留在日志中。
                        _debug_log_gradient_norms(model.model)

                        # 全局梯度范数裁剪：只限制过大的本地梯度更新，
                        # 不改变 lr=0.1，也不修改模型结构或服务器聚合算法。
                        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                            model.model.parameters(),
                            max_norm=DEBUG_GRAD_CLIP_MAX_NORM,
                            error_if_nonfinite=True,
                        )

                        grad_norm_after_clip = _debug_global_gradient_l2(
                            model.model
                        )

                        # 检查裁剪后的梯度幅值，避免用裁剪前的阈值检查
                        # 提前终止本次稳定性验证。
                        _debug_check_gradient_magnitude(
                            model.model,
                            threshold=DEBUG_GRAD_THRESHOLD,
                        )

                        print(
                            "[GRAD-CLIP] {context}; before={before:.8e}, "
                            "after={after:.8e}, max_norm={max_norm:.8e}".format(
                                context=_debug_context_text(),
                                before=float(grad_norm_before_clip.detach().item()),
                                after=grad_norm_after_clip,
                                max_norm=DEBUG_GRAD_CLIP_MAX_NORM,
                            ),
                            flush=True,
                        )

                        model.optimizer.step()

                        _debug_check_parameter_magnitude(
                            model.model,
                            threshold=DEBUG_PARAM_THRESHOLD,
                        )

                    _DEBUG_CONTEXT.update({
                        "seed": seed,
                        "round_iter": num_iter,
                        "client": n,
                        "local_step": "after_local_training",
                    })

                    w_tmp = model.get_weight()
                    w_tmp -= w_global
                    _debug_check_finite("client_model_delta", w_tmp)
                    print(
                        "[CLIENT-DELTA] {context}; max_abs={value:.6e}".format(
                            context=_debug_context_text(),
                            value=_debug_max_abs(w_tmp),
                        ),
                        flush=True,
                    )

                    agg_weight = None

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
                        agg_weight_each_node[n] = agg_weight
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

            _DEBUG_CONTEXT.update({
                "seed": seed,
                "round_iter": num_iter,
                "client": "server",
                "local_step": "before_global_update",
            })
            _debug_check_finite("server_aggregate_update", w_tmp_a)
            print(
                "[SERVER-UPDATE] {context}; aggregate_max_abs={value:.6e}".format(
                    context=_debug_context_text(),
                    value=_debug_max_abs(w_tmp_a),
                ),
                flush=True,
            )

            w_global += torch.tensor(step_size_global).to(device) * w_tmp_a
            _debug_check_finite("global_weights_after_update", w_global)
            print(
                "[GLOBAL-WEIGHTS] {context}; max_abs={value:.6e}".format(
                    context=_debug_context_text(),
                    value=_debug_max_abs(w_global),
                ),
                flush=True,
            )

            num_iter = num_iter + iters_per_round

            if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                model.assign_weight(w_global)
                torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                last_save_checkpoint = num_iter

            if num_iter - last_output >= min_iters_per_eval:
                stat.collect_stat_eval(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                last_output = num_iter

            if num_iter >= max_iter:
                break

        if save_final_model:
            # 保存聚合后的全局模型，而不是最后一个客户端的本地模型
            model.assign_weight(w_global)
            torch.save(model.model.state_dict(), 'final_model_sim' + str(seed) + '_iter' + str(num_iter) + '_' + save_model_file)

        for handle in debug_hook_handles:
            handle.remove()

        del model
        del w_global
        # MoE 双路聚合不使用原始整模型累积变量 w_accumulate
        if weighting_method != 'mifa' and not has_experts:
            del w_accumulate

        torch.cuda.empty_cache()
