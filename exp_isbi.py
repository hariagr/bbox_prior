from train import main as trainfunc
from train import get_args_parser as trainfunc_args_parser
import sys
import numpy as np


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="", add_help=add_help)

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--train-file", default=None, type=str, help="box annotations for training")
    parser.add_argument("--config", default=None, type=str,
                        help="configuration name used to set filename of the csv files")
    parser.add_argument("--tune-batch-size", dest="tune_batch_size", help="", action="store_true")
    parser.add_argument("--baseline", dest="baseline", help="", action="store_true")
    parser.add_argument("--bboxprior-for-classification", dest="bboxprior_for_classification", help="",
                        action="store_true")
    parser.add_argument("--tune-bbp-coverage", dest="tune_bbp_coverage", help="", action="store_true")
    parser.add_argument("--tune-alpha", dest="tune_alpha", help="", action="store_true")
    parser.add_argument("--mean-or-meanIOU", dest="mean_or_meanIOU", help="", action="store_true")
    parser.add_argument("--tune-tauc", dest="tune_tauc", help="", action="store_true")
    parser.add_argument("--prefix", default='usd', type=str, help="prefix for training files")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    return parser


def baseline(args):
    prefix = args.prefix
    wlimages = np.array([10, 50, 90])
    for wl in wlimages:
        train_file = prefix + '_wl' + str(wl) + '.csv'
        config = prefix + '_baseline_wl' + str(wl)
        args = ['--data-path', args.data_path, '--train-file', train_file, '--results-dir', args.results_dir,
                '--config', config,
                '--gt-bbox-loss', args.gt_bbox_loss, '--workers', str(args.workers), '--batch-size',
                str(args.batch_size),
                '--epoch', str(args.epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                '--amp', '--balance', '--device', args.device, '--eval-freq', str(args.eval_freq)]

        old_sys_argv = sys.argv
        sys.argv = [old_sys_argv[0]] + args
        args = trainfunc_args_parser().parse_args()
        training_time = trainfunc(args)
        print(f"config: {config}, batch_size: {args.batch_size}, training time/epoch: {training_time / args.epochs}")


def tune_alpha(args):
    prefix = args.prefix
    bbox_samplings = ['mean', 'mode']

    alphas = np.array([0, 1e-3, 1e-2, 1e-1, 1])
    exp_tcs = np.array([10, 15])
    wlimages = np.array([10, 50, 90])
    ptimages = 100 - wlimages
    cov = 0.0
    alpha_ct = 1.0

    for bbox_sampling in bbox_samplings:
        for exp_tc in exp_tcs:
            for wl, pt in zip(wlimages, ptimages):
                for alpha in alphas:
                    wl_file = prefix + '_wl' + str(wl) + '.csv'
                    pt_file = prefix + '_pt' + str(pt) + '.csv'
                    config = prefix + '_wl' + str(wl) + '_pt' + str(pt) + '_alpha' + str(alpha) + '_st_loss' + str(
                        args.st_bbox_loss) + '_cov' + str(cov) + '_samp_' + str(bbox_sampling) + '_tc' + str(exp_tc)
                    args = ['--data-path', args.data_path, '--train-file', wl_file, '--results-dir', args.results_dir,
                            '--config', config, '--train-points-file', pt_file,
                            '--gt-bbox-loss', args.gt_bbox_loss, '--st-bbox-loss', args.st_bbox_loss,
                            '--workers', str(args.workers), '--batch-size', str(args.batch_size),
                            '--epoch', str(args.epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                            '--amp', '--balance', '--device', args.device, '--eval-freq', str(args.eval_freq),
                            '--alpha-ct', str(alpha_ct), '--alpha', str(alpha), '--bbox-sampling', bbox_sampling,
                            '--bbp-coverage', str(cov), '--bbp-sampling-step', str(0.05), '--exp-tc', str(exp_tc)]
                    old_sys_argv = sys.argv
                    sys.argv = [old_sys_argv[0]] + args
                    args = trainfunc_args_parser().parse_args()
                    training_time = trainfunc(args)
                    print(
                        f"config: {config}, batch_size: {args.batch_size}, training time/epoch: {training_time / args.epochs}")


def mean_or_meanIOU(args):
    prefix = args.prefix
    bbox_samplings = ['mean', 'mean_IOU']

    wlimages = np.array([20])
    ptimages = 100 - wlimages

    for bbox_sampling in bbox_samplings:
        for wl, pt in zip(wlimages, ptimages):
            wl_file = prefix + '_wl' + str(wl) + '.csv'
            pt_file = prefix + '_pt' + str(pt) + '.csv'
            config = prefix + '_wl' + str(wl) + '_pt' + str(pt) + '_' + str(bbox_sampling)
            args = ['--data-path', args.data_path, '--train-file', wl_file, '--train-points-file', pt_file,
                    '--results-dir', args.results_dir, '--config', config,
                    '--gt-bbox-loss', args.gt_bbox_loss, '--st-bbox-loss', args.st_bbox_loss,
                    '--workers', str(args.workers), '--batch-size', str(args.batch_size),
                    '--epoch', str(args.epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                    '--amp', '--device', args.device, '--eval-freq', str(args.eval_freq),
                    '--alpha-ct', str(0.0), '--bbox-sampling', bbox_sampling, '--random-seed', str(args.random_seed)]
            old_sys_argv = sys.argv
            sys.argv = [old_sys_argv[0]] + args
            args = trainfunc_args_parser().parse_args()
            training_time = trainfunc(args)
            print(
                f"config: {config}, batch_size: {args.batch_size}, training time/epoch: {training_time / args.epochs}")

def tune_tauc(args):
    prefix = args.prefix
    bbox_sampling = 'mean_IOU'

    wlimages = np.array([20])
    ptimages = 100 - wlimages

    taucs = np.array([0.2, 0.3, 0.4, 0.5])
    for tauc in taucs:
        for wl, pt in zip(wlimages, ptimages):
            wl_file = prefix + '_wl' + str(wl) + '.csv'
            pt_file = prefix + '_pt' + str(pt) + '.csv'
            config = prefix + '_wl' + str(wl) + '_pt' + str(pt) + '_tauc' + str(tauc)
            args = ['--data-path', args.data_path, '--train-file', wl_file, '--train-points-file', pt_file,
                    '--results-dir', args.results_dir, '--config', config,
                    '--gt-bbox-loss', args.gt_bbox_loss, '--st-bbox-loss', args.st_bbox_loss,
                    '--workers', str(args.workers), '--batch-size', str(args.batch_size),
                    '--epoch', str(args.epochs), '--lr', str(args.lr), '--beta', str(args.beta),
                    '--amp', '--device', args.device, '--eval-freq', str(args.eval_freq),
                    '--alpha-ct', str(0.0), '--bbox-sampling', bbox_sampling, '--random-seed', str(args.random_seed),
                    '--tauc', str(tauc)]
            old_sys_argv = sys.argv
            sys.argv = [old_sys_argv[0]] + args
            args = trainfunc_args_parser().parse_args()
            training_time = trainfunc(args)
            print(
                f"config: {config}, batch_size: {args.batch_size}, training time/epoch: {training_time / args.epochs}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    args.data_path = '../data/USD/'
    args.results_dir = '../results/'
    args.beta = 0.99
    args.gt_bbox_loss = 'l1'
    args.st_bbox_loss = 'l2'
    args.lr = 0.01
    args.epochs = 50
    args.eval_freq = 50
    args.random_seed = 2

    if args.baseline:
        baseline(args)

    if args.tune_alpha:
        tune_alpha(args)

    if args.mean_or_meanIOU:
        mean_or_meanIOU(args)

    if args.tune_tauc:
        tune_tauc(args)
