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

        # Expert activation statistics are used by all activation-frequency schemes.
        # expert_freq_all: both shared and expert parameters use activation counts.
        # shared_avg_expert_freq: shared parameters use FedAvg; experts use
        # per-expert activation-count weights. Neither scheme applies FedAU intervals.
        use_expert_freq_all = (
            weighting_method == 'adaptive'
            and aggregation_scheme == 'expert_freq_all'
        )
        use_shared_avg_expert_freq = (
            weighting_method == 'adaptive'
            and aggregation_scheme == 'shared_avg_expert_freq'
        )
        use_activation_freq_experts = (
            use_expert_freq_all or use_shared_avg_expert_freq
        )
        track_expert_activation = (
            weighting_method == 'adaptive'
            and aggregation_scheme in (
                'shared_au_expert_freq_au',
                'expert_freq_all',
                'shared_avg_expert_freq'
            )
        )

        if (
            aggregation_scheme in (
                'shared_au_expert_freq_au',
                'expert_freq_all',
                'shared_avg_expert_freq'
            )
            and weighting_method != 'adaptive'
        ):
            raise ValueError(
                aggregation_scheme + ' requires -weighting adaptive.'
            )

        expert_masks = None
        expert_activation_log_filename = None

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
                if use_activation_freq_experts:
                    log_columns += ['shared_relative_weight']
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
                'expert_freq_all',
                'shared_avg_expert_freq'
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
                # Shared parameters use original FedAU. Expert parameters
                # also start with FedAU, then receive activation weighting.
                fedavg_mask = torch.zeros_like(expert_mask)
                fedau_mask = torch.ones_like(expert_mask)
            elif aggregation_scheme == 'expert_freq_all':
                # Shared and expert parameter groups are both aggregated
                # directly with activation-count weights below.
                fedavg_mask = shared_mask
                fedau_mask = expert_mask
            elif aggregation_scheme == 'shared_avg_expert_freq':
                # Shared parameters use FedAvg; each expert uses its own
                # activation-count weights in the custom aggregation below.
                fedavg_mask = shared_mask
                fedau_mask = expert_mask
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
                if use_activation_freq_experts:
                    expert_freq_update_sum = torch.zeros_like(w_global)
                    expert_freq_weight_sum = [
                        0.0 for _ in range(model.model.num_experts)
                    ]
                    if use_expert_freq_all:
                        shared_freq_update_sum = torch.zeros_like(w_global)
                        shared_freq_weight_sum = 0.0
                    else:
                        shared_avg_update_sum = torch.zeros_like(w_global)
                        shared_avg_participating_count = 0
                else:
                    fedavg_update_sum = torch.zeros_like(w_global)
                    fedau_update_sum = torch.zeros_like(w_global)
                    participating_count = 0

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
                        if use_activation_freq_experts:
                            # Activation-frequency schemes do not apply the
                            # adaptive participation-interval (FedAU) weight.
                            agg_weight = None
                        elif len(participate_intervals_at_node[n]) > 0:
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

                    if track_expert_activation and not use_activation_freq_experts:
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

                        # Preserve the original scheme: apply each client's
                        # relative expert frequency to that expert's parameters.
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
                        if use_activation_freq_experts:
                            activation_counts = (
                                expert_activation_counts_each_node[n].to(
                                    device=w_global.device,
                                    dtype=w_global.dtype
                                )
                            )

                            if use_expert_freq_all:
                                # Shared parameters use total expert activations.
                                total_activation_count = activation_counts.sum()
                                if total_activation_count.item() > 0:
                                    shared_freq_update_sum[shared_mask] += (
                                        w_tmp[shared_mask] * total_activation_count
                                    )
                                    shared_freq_weight_sum += float(
                                        total_activation_count.item()
                                    )
                            else:
                                # Shared parameters use ordinary FedAvg over
                                # clients that participated in this round.
                                shared_avg_update_sum[shared_mask] += (
                                    w_tmp[shared_mask]
                                )
                                shared_avg_participating_count += 1

                            # Each expert has an independent activation-count
                            # denominator across all participating clients.
                            for expert_idx in range(model.model.num_experts):
                                expert_count = float(
                                    activation_counts[expert_idx].item()
                                )
                                if expert_count > 0:
                                    expert_freq_update_sum[
                                        expert_masks[expert_idx]
                                    ] += (
                                        w_tmp[expert_masks[expert_idx]]
                                        * expert_count
                                    )
                                    expert_freq_weight_sum[expert_idx] += (
                                        expert_count
                                    )
                        else:
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

            # Activation-frequency logging, client by client.
            if track_expert_activation:
                round_expert_totals = torch.stack(
                    expert_activation_counts_each_node
                ).sum(dim=0).float()
                round_total_activations = float(
                    round_expert_totals.sum().item()
                )

                with open(expert_activation_log_filename, 'a') as f:
                    for n in range(n_nodes):
                        counts = expert_activation_counts_each_node[n]
                        total_count = int(counts.sum().item())
                        if total_count > 0:
                            frequencies = counts.float() / total_count
                        else:
                            frequencies = torch.zeros(
                                model.model.num_experts,
                                dtype=torch.float32
                            )

                        if use_activation_freq_experts:
                            # Actual normalized server weight for each expert.
                            relative_weights = torch.where(
                                round_expert_totals > 0,
                                counts.float() / round_expert_totals.clamp(min=1),
                                torch.zeros_like(round_expert_totals)
                            )
                            if use_expert_freq_all:
                                shared_relative_weight = (
                                    total_count / round_total_activations
                                    if round_total_activations > 0 else 0.0
                                )
                            else:
                                round_participating_count = sum(
                                    expert_participated_each_node
                                )
                                shared_relative_weight = (
                                    1.0 / round_participating_count
                                    if expert_participated_each_node[n]
                                    and round_participating_count > 0
                                    else 0.0
                                )
                        else:
                            # Preserve the original scheme's local frequency
                            # multipliers for backward-compatible logging.
                            relative_weights = (
                                model.model.num_experts * frequencies
                            )
                            shared_relative_weight = None

                        row = [
                            seed, num_iter, n,
                            int(expert_participated_each_node[n]),
                            total_count
                        ]
                        row += counts.tolist()
                        row += [float(x) for x in frequencies.tolist()]
                        row += [float(x) for x in relative_weights.tolist()]
                        if use_activation_freq_experts:
                            row += [shared_relative_weight]
                        f.write(','.join(str(x) for x in row) + '\n')

            if save_agg_weight:
                with open(agg_weight_filename, 'a') as f:
                    f.write('\n')
                with open(participation_count_filename, 'a') as f:
                    f.write('\n')

            # Aggregate parameter groups.
            if use_scheme_aggregation:
                w_tmp_a = torch.zeros_like(w_global)

                if use_activation_freq_experts:
                    if use_expert_freq_all:
                        # Shared parameters use the total activation count.
                        if shared_freq_weight_sum > 0:
                            w_tmp_a[shared_mask] = (
                                shared_freq_update_sum[shared_mask]
                                / shared_freq_weight_sum
                            )
                    else:
                        # Shared parameters use FedAvg over participating clients.
                        if shared_avg_participating_count > 0:
                            w_tmp_a[shared_mask] = (
                                shared_avg_update_sum[shared_mask]
                                / shared_avg_participating_count
                            )

                    # Each expert is normalized independently. If no client
                    # activated an expert this round, its update stays zero.
                    for expert_idx in range(model.model.num_experts):
                        expert_weight_sum = expert_freq_weight_sum[expert_idx]
                        if expert_weight_sum > 0:
                            w_tmp_a[expert_masks[expert_idx]] = (
                                expert_freq_update_sum[
                                    expert_masks[expert_idx]
                                ] / expert_weight_sum
                            )
                else:
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
