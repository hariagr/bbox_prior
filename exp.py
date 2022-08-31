from train import main as trainfunc
from train import get_args_parser as trainfunc_args_parser
import sys

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

    return parser

# through this experiment, we intend to find batch size and epoch values
def tune_batch_size(args):
    data_path = '../data/USD/'
    results_dir = '../results/'
    epochs = 80
    lr = 0.01
    workers = args.workers
    beta = 0.99
    device = args.device
    bbox_loss = 'l1'

    # experiment 0: batch size, learning rate,
    batch_sizes = [2, 4, 8, 16]
    train_file = args.train_file  #'train_usd50_wl5.csv'
    config = args.config  #'usd50_wl5_'

    for batch_size in batch_sizes:
        args = ['--data-path', data_path, '--train-file', train_file, '--results-dir', results_dir,
                '--config', config + '_b' + str(batch_size),
                '--bbox-loss', bbox_loss, '--workers', str(workers), '--batch-size', str(batch_size),
                '--epoch', str(epochs), '--lr', str(lr), '--beta', str(beta),
                '--amp', '--balance', '--target-normalization', '--device', device]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {batch_size}, training time/epoch: {training_time / epochs}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    tune_batch_size(args)
