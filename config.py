import os
import torch
import argparse
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str, default='cifar10')  # fashion, svhn, cifar10, cifar100, cinic10
parser.add_argument('-pretrained-model', type=str, default='')
parser.add_argument('-out', type=str, default='results')

parser.add_argument('-weighting', type=str, default='adaptive')  # adaptive, average_participating, average_all, fedvarp, mifa, known_prob
# Router load-balancing regularizer coefficient; 0 disables the extra loss.
parser.add_argument(
    '-router-balance-lambda',
    type=float,
    default=0.0,
    help='Top-2 router load balance coefficient (0.0 disables the regularizer).'
)

# One row per communication round with hard loads, soft probabilities and losses.
parser.add_argument(
    '-router-balance-log',
    type=int,
    choices=[0, 1],
    default=1,
    help='Write a compact CSV of router loads without printing extra console lines.'
)

# Normalize each expert's parameter gradients by B / n_e after backward().
parser.add_argument(
    '-expert-grad-activation-normalize',
    type=int,
    choices=[0, 1],
    default=0,
    help=(
        'Apply per-minibatch expert gradient normalization by '
        'batch_size / Top-k activation count. Requires an expert-frequency '
        'aggregation scheme.'
    )
)

parser.add_argument(
    '-aggregation-scheme',
    type=str,
    default='shared_avg_expert_au',
    choices=[
        'fedavg_all',
        'fedau_all',
        'shared_avg_expert_au',
        'shared_au_expert_avg',
        'shared_au_expert_freq_au',
        'shared_au_expert_freq_scale_norm'
    ],
    help='Parameter-wise aggregation scheme used when -weighting adaptive.'
)
parser.add_argument('-k-adaptive', type=int, default=100000)  # cutoff interval length, only used by the adaptive method
parser.add_argument('-participation', type=str, default='bernoulli')  # bernoulli, markov, cyclic
parser.add_argument('-save-weight-stats', type=int, default=0)

parser.add_argument('-lr', type=float, default=0.1)  # local learning rate of clients
parser.add_argument('-lr-global', type=float, default=1.0)  # global learning rate
parser.add_argument('-minibatch', type=int, default=16)

parser.add_argument('-lr-decay-start-iter', type=int, default=50000)
parser.add_argument('-lr-decay-interval', type=int, default=5000)

parser.add_argument('-iters-total', type=int, default=10000)
parser.add_argument('-seeds', type=str, default='1')  # e.g., 1,2,3
parser.add_argument('-iters-per-round', type=int, default=5)
parser.add_argument('-iters-per-eval', type=int, default=50)
parser.add_argument('-total-workers', type=int, default=250)

parser.add_argument('-gpu', type=int, default=1)  # 1 - use GPU; 0 - do not use GPU
parser.add_argument('-cuda-device', type=int, default=0)

parser.add_argument('-data-dirichlet-alpha', type=float, default=0.1)
parser.add_argument('-participation-dirichlet-alpha', type=float, default=0.1)
parser.add_argument('-participation-prob-mean', type=float, default=0.1)
parser.add_argument('-participation-prob-min', type=float, default=0.02)

parser.add_argument('-save-checkpoint', type=int, default=1)
parser.add_argument('-iters-checkpoint', type=int, default=50000)
parser.add_argument('-save-final-model', type=int, default=1)

args = parser.parse_args()

weighting_method = args.weighting
aggregation_scheme = args.aggregation_scheme

router_balance_lambda = args.router_balance_lambda
router_balance_log = bool(args.router_balance_log)
if router_balance_lambda < 0:
    parser.error('-router-balance-lambda must be nonnegative')

expert_grad_activation_normalize = bool(
    args.expert_grad_activation_normalize
)

k_value = args.k_adaptive
participation_method = args.participation

save_agg_weight = bool(args.save_weight_stats)

use_gpu = bool(args.gpu)
use_gpu = use_gpu and torch.cuda.is_available()
device = (
    torch.device('cuda:' + str(args.cuda_device))
    if use_gpu
    else torch.device('cpu')
)

save_checkpoint = bool(args.save_checkpoint)
iters_checkpoint = args.iters_checkpoint
save_final_model = bool(args.save_final_model)

if args.data == 'fashion':
    dataset = 'FashionMNIST'
    model_name = 'ModelCNNMnist'
elif args.data == 'cifar10':
    dataset = 'CIFAR10'
    model_name = 'ModelCNNCifar10'
elif args.data == 'cifar100':
    dataset = 'CIFAR100'
    model_name = 'ModelCNNCifar100'
elif args.data == 'svhn':
    dataset = 'SVHN'
    model_name = 'ModelCNNSvhn'
elif args.data == 'cinic10':
    dataset = 'CINIC10'
    model_name = 'ModelCNNCinic10'
else:
    raise Exception('Unknown data name')

if expert_grad_activation_normalize:
    if model_name != 'ModelCNNCifar10':
        parser.error(
            'Expert gradient activation normalization is currently '
            'implemented only for ModelCNNCifar10.'
        )
    if (
        weighting_method != 'adaptive'
        or aggregation_scheme not in (
            'shared_au_expert_freq_au',
            'shared_au_expert_freq_scale_norm'
        )
    ):
        parser.error(
            'Expert gradient activation normalization requires '
            '-weighting adaptive and an expert-frequency aggregation scheme.'
        )

max_iter = args.iters_total
simulations_str = args.seeds.split(',')
simulations = [int(i) for i in simulations_str]

dataset_file_path = '/home/lw/Project/dataset'

data_dirichlet_alpha = args.data_dirichlet_alpha
participation_dirichlet_alpha = args.participation_dirichlet_alpha
participation_prob_mean = args.participation_prob_mean
participation_prob_min = args.participation_prob_min

n_nodes = args.total_workers
step_size_local_config = args.lr
step_size_global = args.lr_global

lr_decay_start_iter = args.lr_decay_start_iter
lr_decay_interval = args.lr_decay_interval

batch_size_train = args.minibatch
batch_size_eval = 256
iters_per_round = args.iters_per_round
min_iters_per_eval = args.iters_per_eval

if weighting_method == 'adaptive':
    results_file_prefix = (
        args.out
        + '_' + dataset
        + '_' + model_name
        + '_' + weighting_method
        + '_' + participation_method
        + '_lr' + str(step_size_local_config)
        + '_lr_global' + str(step_size_global)
        + '_dataAlpha' + str(data_dirichlet_alpha)
        + '_partAlpha' + str(participation_dirichlet_alpha)
        + '_kValue' + str(k_value)
    )
else:
    results_file_prefix = (
        args.out
        + '_' + dataset
        + '_' + model_name
        + '_' + weighting_method
        + '_' + participation_method
        + '_lr' + str(step_size_local_config)
        + '_lr_global' + str(step_size_global)
        + '_dataAlpha' + str(data_dirichlet_alpha)
        + '_partAlpha' + str(participation_dirichlet_alpha)
    )

results_file_prefix += '_scheme_' + aggregation_scheme
# Keep results from LB-on and LB-off experiments separate even if -out is reused.
if router_balance_lambda > 0:
    results_file_prefix += '_routerLB' + str(router_balance_lambda)

if expert_grad_activation_normalize:
    results_file_prefix += '_expertGradActNorm'

save_model_file = results_file_prefix + '.model'

if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10' or dataset == 'CINIC10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    transform_train_eval = None
elif dataset == 'SVHN':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
    ])
    transform_train_eval = None
elif dataset == 'CIFAR100':
    transform_train = transforms.Compose([
        transforms.ConvertImageDtype(torch.uint8),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ColorJitter(brightness=0.05, hue=0.05),
        transforms.RandomPosterize(bits=2),
        transforms.RandomEqualize(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    transform_train_eval = transforms.Compose([
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])
else:
    transform_train = None
    transform_train_eval = None
