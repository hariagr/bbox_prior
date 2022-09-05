from train import main as trainfunc
from train import get_args_parser as trainfunc_args_parser
import sys
import numpy as np

# experiment 1 - Baseline (All boxes) -- upper bound
# sub experiments (a) GT_loss: L1, L2, smooth-L1
#                 (b) beta (class imbalance parameter)

# experiment 2 - Pruned dataset (pick GT_loss function from exp 1)
# sub experiments (a) beta

# experiment 3 - consider points, but alpha = 0 (bbox_prior only for classification)
#                pick GT_loss and beta from exp 1 (the number of class instances are the same
# sub experiments (a) pruning
#                 (b) bbp_coverage, bbp_sampling_step

# experiment 4 - points + alpha > 0
# sub experiments: at different pruning
#                  (a) alpha
#                  (b) ST_loss
#                  (c) bbp_coverage, bbp_sampling_step

# experiments 5 - 100% points, pick beta from exp 1, alpha = 1
# sub experiments: (a) ST_loss
#                  (b) bbp_coverage, bbp_sampling_step

######### parameters to set ############
# data_path, train_file, train_points_file, results_dir
# batch_size, epochs, workers, random_seed
# amp, balance, target_normalization,
# beta, alpha, bbp_coverage, bbp_sampling_step, GT_bbox_loss, ST_bbox_loss

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--train-file", default=None, type=str, help="box annotations for training")
    parser.add_argument("--config", default=None, type=str,
                        help="configuration name used to set filename of the csv files")
    parser.add_argument("--tune-batch-size", dest="tune_batch_size", help="", action="store_true")
    parser.add_argument("--baseline", dest="baseline", help="", action="store_true")
    parser.add_argument("--bboxprior-for-classification", dest="bboxprior_for_classification", help="", action="store_true")
    return parser

# through this experiment, we intend to find batch size and epoch values
def tune_batch_size(args):
    epochs = 80
    workers = args.workers
    device = args.device

    # experiment 0: batch size, learning rate,
    batch_sizes = [2, 4, 8, 16]
    train_file = args.train_file  #'train_usd50_wl5.csv'
    config = args.config  #'usd50_wl5_'

    for batch_size in batch_sizes:
        args = ['--data-path', args.data_path, '--train-file', train_file, '--results-dir', args.results_dir,
                '--config', config + '_b' + str(batch_size),
                '--bbox-loss', args.bbox_loss, '--workers', str(workers), '--batch-size', str(batch_size),
                '--epoch', str(epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                '--amp', '--balance', '--target-normalization', '--device', device]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {batch_size}, training time/epoch: {training_time / epochs}")

def baseline(args):
    epochs = 50
    batch_size = 8
    workers = args.workers
    device = args.device

    wlimages = np.array([5, 20, 40, 60, 80])
    for wl in wlimages:
        train_file = 'train_usd50_wl' + str(wl) + '.csv'
        config = 'baseline_usd50_wl' + str(wl)
        args = ['--data-path', args.data_path, '--train-file', train_file, '--results-dir', args.results_dir,
                '--config', config + '_b' + str(batch_size),
                '--bbox-loss', args.bbox_loss, '--workers', str(workers), '--batch-size', str(batch_size),
                '--epoch', str(epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                '--amp', '--balance', '--target-normalization', '--device', device, '--eval-freq', str(epochs)]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {batch_size}, training time/epoch: {training_time / epochs}")

def bboxprior_for_classification(args):
    epochs = 50
    batch_size = 8
    workers = args.workers
    device = args.device

    wlimages = np.array([5, 20, 40, 60, 80])
    for wl in wlimages:
        wl_file = 'train_usd50_wl' + str(wl) + '.csv'
        pt_file = 'train_usd50_pt' + str(100 - wl) + '.csv'

        config = 'bp_for_cls_usd50_wl' + str(wl) + '_pt' + str(100 - wl)
        args = ['--data-path', args.data_path, '--train-file', wl_file, '--results-dir', args.results_dir,
                '--config', config, '--train-points-file', pt_file,
                '--bbox-loss', args.bbox_loss, '--workers', str(workers), '--batch-size', str(batch_size),
                '--epoch', str(epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                '--amp', '--balance', '--target-normalization', '--device', device, '--eval-freq', str(epochs),
                '--alpha', str(0), '--bbp-coverage', str(0.25), '--bbp-sampling-step', str(0.05)]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {batch_size}, training time/epoch: {training_time / epochs}")

    wl = 5
    ptimages = np.array([10, 30, 50, 70])
    for pt in ptimages:
        wl_file = 'train_usd50_wl' + str(wl) + '.csv'
        pt_file = 'train_usd50_pt' + str(pt) + '.csv'

        config = 'bp_for_cls_usd50_wl' + str(wl) + '_pt' + str(pt)
        args = ['--data-path', args.data_path, '--train-file', wl_file, '--results-dir', args.results_dir,
                '--config', config, '--train-points-file', pt_file,
                '--bbox-loss', args.bbox_loss, '--workers', str(workers), '--batch-size', str(batch_size),
                '--epoch', str(epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                '--amp', '--balance', '--target-normalization', '--device', device, '--eval-freq', str(epochs),
                '--alpha', str(0), '--bbp-coverage', str(0.25), '--bbp-sampling-step', str(0.05)]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {batch_size}, training time/epoch: {training_time / epochs}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    args.data_path = '../data/USD/'
    args.results_dir = '../results/'
    args.beta = 0.99
    args.bbox_loss = 'l1'
    args.lr = 0.01

    if args.tune_batch_size:
        tune_batch_size(args)

    if args.baseline:
        baseline(args)

    if args.bboxprior_for_classification:
        bboxprior_for_classification(args)






