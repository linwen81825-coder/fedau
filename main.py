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


def normalize_expert_gradients_by_activation(model, batch_size):
    """
    Normalize each expert's parameter gradients by its Top-k activation count
    in the current minibatch.

    With a mean-reduced batch loss:
        grad_e = (1 / B) * sum_{b: e selected} grad_{b,e}

    Multiplying by B / n_e changes this to the average gradient over samples
    that selected expert e. Router weights remain inside each sample gradient.
    """
    if not hasattr(model, 'experts') or not hasattr(model, 'last_expert_counts'):
        raise ValueError(
            'Activation-count gradient normalization requires a sparse MoE '
            'model exposing experts and last_expert_counts.'
        )

    counts = model.last_expert_counts.detach()
    if counts.numel() != len(model.experts):
        raise ValueError(
            'last_expert_counts does not match the number of experts.'
        )

    # An unselected expert has no gradient (grad is None). clamp_min(1)
    # prevents division by zero; the scale for an unselected expert is unused.
    scales = (
        float(batch_size)
        / counts.clamp_min(1).to(dtype=torch.float32)
    )

    for expert_idx, expert in enumerate(model.experts):
        scale = scales[expert_idx]
        for parameter in expert.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(dtype=parameter.grad.dtype))

    
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
            f.write('sim_seed,num_iter,aggregation_weight_each_node\n')

        participation_count_filename = (
            'participationCount_' + results_file_prefix + '.csv'
        )
        with open(participation_count_filename, 'a') as f:
            f.write('sim_seed,num_iter,participation_count_each_node\n')

    # CSV for all selected clients' Top-2 load, full router probability
    # and averaged classification / auxiliary losses, per communication round.
    router_balance_logging = (
        router_balance_log and model_name == 'ModelCNNCifar10'
    )
    if router_balance_lambda > 0 and model_name != 'ModelCNNCifar10':
        raise ValueError(
            'Router balance is implemented only for ModelCNNCifar10.'
        )

    if router_balance_logging:
        router_balance_csv = (
            'routerBalanceStats_' + results_file_prefix + '.csv'
        )
        if (not os.path.exists(router_balance_csv)
                or os.path.getsize(router_balance_csv) == 0):
            with open(router_balance_csv, 'a', encoding='utf-8') as f:
                f.write(
                    'sim_seed,num_iter,participating_clients,samples,'
                    'expert_0_count,expert_1_count,expert_2_count,expert_3_count,'
                    'expert_0_frequency,expert_1_frequency,'
                    'expert_2_frequency,expert_3_frequency,'
                    'expert_0_probability,expert_1_probability,'
                    'expert_2_probability,expert_3_probability,'
                    'mean_classification_loss,mean_balance_loss,lambda\n'
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
                images.append(train_loader_list[n].dataset[i][0])
                l = train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)
            return torch.stack(images), torch.stack(labels)

        # Global model parameters.
        w_global = model.get_weight()

        # Split expert and shared parameters.
        expert_mask = model.get_expert_weight_mask()
        shared_mask = ~expert_mask
        assert expert_mask.numel() == w_global.numel(), \
            "Expert mask dimension does not match model weights!"

        # Both expert-frequency variants share the same activation counts.
        expert_frequency_schemes = (
            'shared_au_expert_freq_au',
            'shared_au_expert_freq_scale_norm',
        )
        scale_preserving_expert_freq = (
            aggregation_scheme == 'shared_au_expert_freq_scale_norm'
        )
        track_expert_activation = (
            weighting_method == 'adaptive'
            and aggregation_scheme in expert_frequency_schemes
        )

        if (
            aggregation_scheme in expert_frequency_schemes
            and weighting_method != 'adaptive'
        ):
            raise ValueError(
                'Expert-frequency aggregation requires -weighting adaptive.'
            )

        if expert_grad_activation_normalize and not track_expert_activation:
            raise ValueError(
                'Expert gradient activation normalization requires '
                'shared_au_expert_freq_au or '
                'shared_au_expert_freq_scale_norm with -weighting adaptive.'
            )

        expert_masks = None
        expert_activation_log_filename = None

        # Optional diagnostics: one summary row per communication round.
        scale_norm_log_filename = None
        if scale_preserving_expert_freq:
            scale_norm_log_filename = (
                'expertScaleNormStats_' + results_file_prefix + '.csv'
            )
            if (
                not os.path.exists(scale_norm_log_filename)
                or os.path.getsize(scale_norm_log_filename) == 0
            ):
                with open(scale_norm_log_filename, 'a', encoding='utf-8') as f:
                    columns = [
                        'sim_seed', 'num_iter', 'participating_clients',
                        'sum_fedau_weights',
                    ]
                    columns += [
                        f'expert_{e}_sum_fedau_times_freq' for e in range(4)
                    ]
                    columns += [
                        f'expert_{e}_normalization_factor' for e in range(4)
                    ]
                    f.write(','.join(columns) + '\n')

        if track_expert_activation:
            if not hasattr(model.model, 'num_experts'):
                raise ValueError(
                    'Expert-frequency aggregation requires a MoE model '
                    'with num_experts.'
                )

            expert_masks = model.get_expert_weight_masks()
            if len(expert_masks) != model.model.num_experts:
                raise ValueError('The number of expert masks is incorrect.')

            for mask in expert_masks:
                if mask.numel() != w_global.numel():
                    raise ValueError(
                        'Expert mask dimension does not match model weights!'
                    )

            expert_activation_log_filename = (
                'expertActivationCounts_' + results_file_prefix + '.csv'
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

        # Parameter-wise aggregation masks.
        use_scheme_aggregation = (weighting_method == 'adaptive')
        if use_scheme_aggregation:
            has_expert = torch.any(expert_mask).item()
            has_shared = torch.any(shared_mask).item()

            if aggregation_scheme in (
                'shared_avg_expert_au',
                'shared_au_expert_avg',
                'shared_au_expert_freq_au',
                'shared_au_expert_freq_scale_norm'
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
            elif aggregation_scheme in expert_frequency_schemes:
                # Shared parameters always use original FedAU.
                # Expert parameters additionally use client activation frequency;
                # the scale-normalized variant restores original FedAU mass.
                fedavg_mask = torch.zeros_like(expert_mask)
                fedau_mask = torch.ones_like(expert_mask)
            else:
                raise ValueError(
                    'Unknown aggregation scheme: ' + str(aggregation_scheme)
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
            update_all_avg = torch.zeros(w_global.shape[0]).to('cpu')

        learning_rate_decay_level = 0

        while True:

            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))
                with open(participation_count_filename, 'a') as f:
                    f.write(str(seed) + ',' + str(num_iter))

            # Keep the original learning-rate decay logic.
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

            accumulated = 0

            if router_balance_logging:
                round_router_counts = torch.zeros(4, dtype=torch.long, device=device)
                round_router_prob_sums = torch.zeros(4, dtype=torch.float64, device=device)
                round_router_samples = 0
                round_router_participants = 0
                round_ce_loss = torch.zeros((), dtype=torch.float32, device=device)
                round_lb_loss = torch.zeros((), dtype=torch.float32, device=device)
                round_batches = 0

            if use_scheme_aggregation:
                fedavg_update_sum = torch.zeros_like(w_global)
                fedau_update_sum = torch.zeros_like(w_global)
                participating_count = 0

            if scale_preserving_expert_freq:
                # A = sum_{i in S_t} a_i, and B_e = sum_i a_i * r_{i,e}.
                # The weighted expert deltas are already added to
                # fedau_update_sum below; normalize them after all clients.
                round_fedau_weight_sum = 0.0
                round_expert_weighted_mass = torch.zeros(
                    model.model.num_experts,
                    dtype=torch.float64,
                    device=device
                )

            if track_expert_activation:
                expert_activation_counts_each_node = [
                    torch.zeros(model.model.num_experts, dtype=torch.long)
                    for _ in range(n_nodes)
                ]
                expert_participated_each_node = [False for _ in range(n_nodes)]

            # Local training on sampled clients.
            for n in range(n_nodes):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participated = True
                    model.assign_weight(w_global)
                    model.model.train()

                    if router_balance_logging:
                        round_router_participants += 1

                    # ------------------------------------------------
                    if track_expert_activation:
                        local_expert_activation_counts = torch.zeros(
                            model.model.num_experts,
                            dtype=torch.long,
                            device=device
                        )

                    for i in range(0, iters_per_round):
                        images, labels = sample_minibatch(n)
                        images = images.to(device)
                        labels = labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous()

                        model.optimizer.zero_grad()
                        output = model.model(images)

                        if track_expert_activation:
                            local_expert_activation_counts += (
                                model.model.last_expert_counts.detach()
                            )

                        classification_loss = model.loss_fn(output, labels)
                        loss = classification_loss

                        # Add the Router regularizer to the LOCAL loss.
                        # No changes to FedAU weighting or server aggregation.
                        if router_balance_lambda > 0:
                            loss = loss + (
                                router_balance_lambda
                                * model.model.last_router_balance_loss
                            )

                        if router_balance_logging:
                            # Hard counts: two selected experts per sample.
                            round_router_counts += (
                                model.model.last_expert_counts.detach()
                            )
                            batch_size = int(images.size(0))
                            round_router_prob_sums += (
                                model.model.last_router_mean_probs.detach()
                                .to(dtype=torch.float64) * batch_size
                            )
                            round_router_samples += batch_size
                            round_ce_loss += classification_loss.detach()
                            round_lb_loss += (
                                model.model.last_router_balance_loss.detach()
                            )
                            round_batches += 1

                        loss.backward()

                        # Normalize ONLY expert parameter gradients.
                        # For mean-reduced CE, multiplying by B / n_e turns
                        # the batch-mean gradient into the mean over samples
                        # that selected expert e. Router weights alpha_{b,e}
                        # remain part of each sample's gradient.
                        if expert_grad_activation_normalize:
                            normalize_expert_gradients_by_activation(
                                model.model,
                                int(images.size(0))
                            )

                        model.optimizer.step()

                    if track_expert_activation:
                        expert_activation_counts_each_node[n] = (
                            local_expert_activation_counts.detach().cpu()
                        )
                        expert_participated_each_node[n] = True

                    # Local model update.
                    w_tmp = model.get_weight()
                    w_tmp -= w_global
                    agg_weight = None

                    if weighting_method == 'known_prob':
                        w_tmp /= worker_sampler.participation_prob
                        agg_weight = 1 / worker_sampler.participation_prob

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

                    if track_expert_activation:
                        total_activation_count = (
                            local_expert_activation_counts.sum()
                        )
                        if total_activation_count.item() > 0:
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

                        # a_i is the actual FedAU coefficient used above:
                        # 1 for the first participation (no interval history),
                        # otherwise mean observed participation interval.
                        if scale_preserving_expert_freq:
                            effective_fedau_weight = (
                                float(agg_weight)
                                if agg_weight is not None else 1.0
                            )
                            round_fedau_weight_sum += effective_fedau_weight
                            round_expert_weighted_mass += (
                                effective_fedau_weight
                                * expert_frequency.detach().to(torch.float64)
                            )

                        # All expert deltas first receive a_i * r_{i,e}.
                        # In the scale-preserving scheme, multiply the round
                        # expert sum by A / B_e at the aggregation stage.
                        for expert_idx in range(model.model.num_experts):
                            w_tmp[expert_masks[expert_idx]] *= (
                                expert_frequency[expert_idx]
                            )

                    if save_agg_weight:
                        agg_weight_each_node[n] = agg_weight
                        participation_count_each_node[n] += 1

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

                    if track_expert_activation:
                        expert_activation_counts_each_node[n] = (
                            torch.zeros(
                                model.model.num_experts,
                                dtype=torch.long
                            )
                        )

                if save_agg_weight:
                    with open(agg_weight_filename, 'a') as f:
                        if agg_weight_each_node[n] is not None:
                            f.write(',' + "%.4f" % agg_weight_each_node[n])
                        else:
                            f.write(',nan')
                    with open(participation_count_filename, 'a') as f:
                        f.write(',' + str(participation_count_each_node[n]))

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

            # Original activation-frequency logging, client by client.
            if track_expert_activation:
                with open(expert_activation_log_filename, 'a') as f:
                    for n in range(n_nodes):
                        counts = expert_activation_counts_each_node[n]
                        total_count = int(counts.sum().item())
                        if total_count > 0:
                            frequencies = counts.float() / total_count
                            relative_weights = (
                                model.model.num_experts * frequencies
                            )
                        else:
                            frequencies = torch.zeros(
                                model.model.num_experts,
                                dtype=torch.float32
                            )
                            relative_weights = torch.zeros_like(frequencies)
                        row = [
                            seed, num_iter, n,
                            int(expert_participated_each_node[n]),
                            total_count
                        ]
                        row += counts.tolist()
                        row += [float(x) for x in frequencies.tolist()]
                        row += [float(x) for x in relative_weights.tolist()]
                        f.write(','.join(str(x) for x in row) + '\n')

            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write('\n')
                with open(participation_count_filename, 'a') as f:
                    f.write('\n')

            # Scale-preserving frequency weighting, for each expert e:
            #   W_{i,e} = a_i * r_{i,e} / B_e * A
            # where A = sum_{j in S_t} a_j, B_e = sum_{j in S_t} a_j*r_{j,e}.
            # The original FedAU denominator n_nodes is preserved below.
            scale_norm_factors = None
            if scale_preserving_expert_freq:
                scale_norm_factors = []
                for expert_idx in range(model.model.num_experts):
                    expert_mass = float(
                        round_expert_weighted_mass[expert_idx].item()
                    )
                    if expert_mass > 0.0:
                        scale_factor = round_fedau_weight_sum / expert_mass
                        fedau_update_sum[expert_masks[expert_idx]] *= (
                            scale_factor
                        )
                    else:
                        # No client used this expert in this round: B_e = 0.
                        # There is no supported expert delta to aggregate.
                        scale_factor = 0.0
                    scale_norm_factors.append(scale_factor)

            # Aggregate parameter groups.
            if use_scheme_aggregation:
                w_tmp_a = torch.zeros_like(w_global)

                if participating_count > 0:
                    w_tmp_a[fedavg_mask] = (
                        fedavg_update_sum[fedavg_mask]
                        / participating_count
                    )

                # Original Adaptive FedAU denominator (all clients).
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
                        update_all_avg = torch.mean(update_per_node, 0)
                else:
                    w_tmp_a = torch.zeros(w_global.shape[0]).to(device)
            else:
                w_tmp_a = torch.mean(update_per_node, 0).to(device)

            # Update global model.
            w_global += (
                torch.tensor(step_size_global).to(device)
                * w_tmp_a
            )

            num_iter = num_iter + iters_per_round

            if scale_preserving_expert_freq:
                with open(scale_norm_log_filename, 'a', encoding='utf-8') as f:
                    row = [
                        seed, num_iter, participating_count,
                        round_fedau_weight_sum,
                    ]
                    row += round_expert_weighted_mass.tolist()
                    row += scale_norm_factors
                    f.write(','.join(str(x) for x in row) + '\n')

            if router_balance_logging:
                # Explicitly record zero values if no client participated.
                total_slots = int(round_router_counts.sum().item())
                if total_slots:
                    frequencies = (
                        round_router_counts.to(dtype=torch.float64)
                        / total_slots
                    ).tolist()
                else:
                    frequencies = [0.0] * 4
                if round_router_samples:
                    mean_probs = (
                        round_router_prob_sums / round_router_samples
                    ).tolist()
                else:
                    mean_probs = [0.0] * 4
                n_batches = max(1, round_batches)
                row = [
                    seed, num_iter, round_router_participants,
                    round_router_samples,
                    *round_router_counts.tolist(), *frequencies, *mean_probs,
                    float((round_ce_loss / n_batches).item()),
                    float((round_lb_loss / n_batches).item()),
                    router_balance_lambda,
                ]
                with open(router_balance_csv, 'a', encoding='utf-8') as f:
                    f.write(','.join(str(x) for x in row) + '\n')

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

            if num_iter >= max_iter:
                break

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
