r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from cellDataset import CSVDataset
from my_retinanet import retinanet_resnet50_fpn
import transforms as T
from eval_mAP_F1 import evaluate as eval_mAP_F1
from target_normalization import cal_tnorm_weights
from bbox_priors import cal_bbox_priors
from cal_bbox_prior_hp import cal_bbox_prior_hp

import platform

if not platform.system().lower().startswith('dar'):
    import nvidia_dlprof_pytorch_nvtx
    nvidia_dlprof_pytorch_nvtx.init()

try:
    from torchvision import prototype
except ImportError:
    prototype = None

torch.backends.cudnn.benchmark = True

def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(args.data_augmentation)
    elif not args.prototype:
        return presets.DetectionPresetEval()
    else:
        if args.weights:
            weights = prototype.models.get_weight(args.weights)
            return weights.transforms()
        else:
            return prototype.transforms.CocoEval()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="../data/UMID/", type=str, help="dataset path")
    #parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--train-file", default=None, type=str, help="box annotations for training")
    parser.add_argument("--train-points-file", default=None, type=str, help="point annotations for training")
    parser.add_argument("--val-file", default="val.csv", type=str, help="annotations for validation")
    parser.add_argument("--test-file", default="test.csv", type=str, help="annotations for validation")

    #parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler", default="reducelronplateau", type=str, help="name of lr scheduler (default: reducelronplateau)"
    )
    #parser.add_argument(
    #    "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    #)
    #parser.add_argument(
    #    "--lr-steps",
    #    default=[16, 22],
    #    nargs="+",
    #    type=int,
    #    help="decrease lr every step-size epochs (multisteplr scheduler only)",
    #)
    #parser.add_argument(
    #    "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    #)
    parser.add_argument("--random-seed", default=0, type=int, help="random seed for reproducibility")
    parser.add_argument("--eval-freq", default=10, type=int, help="evaluation frequency")
    parser.add_argument("--print-freq", default=1000, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument("--config", default=None, type=str, help="configuration name used to set filename of the csv files")
    parser.add_argument("--results-dir", default=None, type=str, help="path to save csv result files")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int)
    #parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    #parser.add_argument(
    #    "--data-augmentation", default="None", type=str, help="data augmentation policy (default: hflip)"
    #)
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Prototype models only
    parser.add_argument(
        "--prototype",
        dest="prototype",
        help="Use prototype model builders instead those from main area",
        action="store_true",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # balance class frequency in the loss function
    parser.add_argument("--balance", action="store_true", help="handle class imbalance problem")
    parser.add_argument("--beta", default=0.999, type=float, help="parameter for balancing function")

    # target normalization
    parser.add_argument(
        "--target-normalization",
        action="store_true",
        help="normalize the bounding box offset such that std(offset) = 1",
    )

    # freeze batch norm
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="freeze the batch norm in the pretrained model",
    )

    # parameters for bounding box prior strategy
    parser.add_argument("--alpha", default=0, type=float, help="a parameter to weigh stochastic boxes loss function")
    parser.add_argument("--bbp-coverage", default=2, type=int,
                        help="(in terms of std.dev.) - maximum wideness of a stochastic box")
    parser.add_argument("--bbp-sampling-step", default=0.2, type=float,
                        help="sampling of stochastic box wideness")
    parser.add_argument("--bbox-loss", default='smooth_l1', type=str, help="bounding box regression loss function")

    #argprof
    parser.add_argument("--DLprof", action="store_true", help="flag to run profiling")

    return parser


def main(args):
    if args.prototype and prototype is None:
        raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
    if not args.prototype and args.weights:
        raise ValueError("The weights parameter works only in prototype mode. Please pass the --prototype argument.")
    if args.output_dir is not None:
        utils.mkdir(args.output_dir)
    if args.results_dir is not None:
        utils.mkdir(args.results_dir)

    print("Random seed ",args.random_seed)
    utils.seed_everything(args.random_seed)
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    #dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path)
    #dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)

    # dataset = PennFudanDataset(args.data_path,
    #               transforms=T.Compose([
	# 	                T.ToTensor()]))
    # dataset_test = PennFudanDataset(args.data_path)

    annotations_path = os.path.join(args.data_path + 'annotations')
    print("Initializing training and validation dataset classes")
    if args.train_file is not None:
        train_boxes_file = os.path.join(annotations_path, args.train_file)
    else:
        train_boxes_file = None
    if args.train_points_file is not None:
        train_points_file = os.path.join(annotations_path, args.train_points_file)
    else:
        train_points_file = None
    dataset = CSVDataset(args.data_path, train_boxes_file, points_file=train_points_file, beta=args.beta, transform=T.Compose([T.ToTensor()]))
    dataset_val = CSVDataset(args.data_path, os.path.join(annotations_path, args.val_file), transform=T.Compose([T.ToTensor()]))
    dataset_test = CSVDataset(args.data_path, os.path.join(annotations_path, args.test_file), transform=T.Compose([T.ToTensor()]))
    num_classes = dataset.num_classes()

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    if args.balance:
        bl_weights = dataset.bl_weights.to(device)
        print(f"Setting weights {bl_weights} for handling class imbalance problem")
    else:
        bl_weights = (1/num_classes)*torch.ones(num_classes).to(device)

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers, "bl_weights": bl_weights,
              "alpha": args.alpha, "bbp_coverage": args.bbp_coverage, "bbp_sampling_step": args.bbp_sampling_step, "bbox_loss": args.bbox_loss}
    model = retinanet_resnet50_fpn(pretrained=args.pretrained, num_classes=num_classes, freeze_bn=args.freeze_bn, **kwargs)
    model.to(device)

    if args.train_points_file is not None:
        print('Calculating box priors')
        if args.train_file is None:
            model = cal_bbox_priors(model, data_loader_val, device)
        else:
            model = cal_bbox_priors(model, data_loader, device)

    if args.bbp_sampling_step == -1:
        print('Calculating bbox prior hyperparameter')
        model = cal_bbox_prior_hp(model, data_loader, device)

    if args.target_normalization:  # should be on training dataset to consider stochastic boxes
        print('Calculating target normalization weights')
        model = cal_tnorm_weights(model, data_loader, device)

    # normalized targets std. dev. (i.e. label std./pre-normalized target std. dev)
    if args.train_points_file is not None:
        tstd = 1/torch.as_tensor(model.box_coder.weights)
        model.bbox_priors["target_width_std"] = model.bbox_priors["logOfwidth_std"]/tstd[2]
        model.bbox_priors["target_height_std"] = model.bbox_priors["logOfwidth_std"]/tstd[3]
        model.head.bbox_priors = model.bbox_priors
        model.head.regression_head.bbox_priors = model.bbox_priors

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'reducelronplateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.9)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only ReduceLROnPlateau, MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.DLprof:
            with torch.autograd.profiler.emit_nvtx():
                metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq,
                                                scaler)
        else:
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq,
                                            scaler)
        if args.lr_scheduler == 'reducelronplateau':
            lr_scheduler.step(metric_logger.meters.get('loss').value)
        else:
            lr_scheduler.step()
        if args.output_dir is not None and (epoch + 1) % args.eval_freq == 0:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every epoch
        if (epoch + 1) % args.eval_freq == 0:
            # coco_evaluator = evaluate(model, data_loader_val, device=device)  # coco evaluation
            eval_val, eval_time, analysis_table = eval_mAP_F1(dataset_val, model, count=epoch,
                                                              missedLabels=True)  # our evaluation
            # evaluate(model, data_loader_test, device=device)
            eval_test, eval_time, at = eval_mAP_F1(dataset_test, model, count=epoch, missedLabels=True)

            if args.results_dir is not None:
                #filename = os.path.join(args.results_dir, args.config + '_val_' + str(epoch) + '.csv')
                #analysis_table.to_csv(filename, mode='a', header=not os.path.exists(filename))

                eval_val['epoch'] = epoch
                eval_test['epoch'] = epoch

                filename = os.path.join(args.results_dir, args.config + '_val_' + time.strftime('%Y%m%d_%H%M%S',
                                                                                                time.localtime(
                                                                                                    start_time)) + '.csv')
                eval_val.to_csv(filename, mode='a', header=not os.path.exists(filename))

                filename = os.path.join(args.results_dir, args.config + '_test_' + time.strftime('%Y%m%d_%H%M%S',
                                                                                                 time.localtime(
                                                                                                     start_time)) + '.csv')
                eval_test.to_csv(filename, mode='a', header=not os.path.exists(filename))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
