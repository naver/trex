# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import datetime
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import multi_crop
import utils
from trex import tReX


def get_args():
    parser = argparse.ArgumentParser(description="Training t-ReX models on ImageNet-1K")

    # Model parameters
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        choices=["resnet50"],
        help="Name of the architecture to train.",
    )

    # Multi-crop arguments
    parser.add_argument(
        "--mc_global_number",
        type=int,
        default=1,
        help="Number of random global crops.",
    )
    parser.add_argument(
        "--mc_global_scale",
        type=float,
        nargs="+",
        default=(0.25, 1.0),
        help="Scale range for global crops.",
    )
    parser.add_argument(
        "--mc_local_number",
        type=int,
        default=8,
        help="Number of random local crops.",
    )
    parser.add_argument(
        "--mc_local_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.25),
        help="Scale range for local crops.",
    )
    # Projector head arguments
    parser.add_argument(
        "--pr_hidden_layers",
        default=3,
        type=int,
        help="Number of hidden layers in the projector head.",
    )
    parser.add_argument(
        "--pr_hidden_dim",
        default=2048,
        type=int,
        help="Number of hidden units in the hidden layers of the projector head.",
    )
    parser.add_argument(
        "--pr_bottleneck_dim",
        default=256,
        type=int,
        help="Bottleneck layer dimension in the projector head.",
    )
    parser.add_argument(
        "--pr_no_input_l2_norm",
        action="store_true",
        help="Whether to NOT use l2 normalization in the input of the projector head.",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    # Supervised classification parameters
    parser.add_argument(
        "--clf_tau",
        default=0.1,
        type=float,
        help="Temperature for cosine softmax loss.",
    )

    # Memory parameters
    parser.add_argument(
        "--memory_size",
        default=8192,
        type=int,
        help="Size of the memory bank used to compute class prototypes.",
    )

    # Misc
    parser.add_argument(
        "--data_dir",
        default="/path/to/imagenet",
        type=str,
        help="Path to the ImageNet dataset containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=20,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    args = parser.parse_args()

    return args


def main(args):
    utils.init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)
    utils.print_program_info(args, os.path.join(args.output_dir, "program_info.txt"))

    utils.fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data
    # ==================================================
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=multi_crop.MultiCropAugmentation(
            args.mc_global_number,
            args.mc_global_scale,
            args.mc_local_number,
            args.mc_local_scale,
        ),
    )
    print("=> Training dataset:\n{}".format(train_dataset))

    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    print("=> Validation dataset: {}".format(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.DistributedSampler(train_dataset, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # ==================================================
    # Model and optimizer
    # ==================================================
    print("=> creating model '{}'".format(args.arch))
    model = tReX(
        args.arch,
        not args.pr_no_input_l2_norm,
        args.pr_hidden_layers,
        args.pr_hidden_dim,
        args.pr_bottleneck_dim,
        clf_tau=args.clf_tau,
        memory_size=args.memory_size,
    )
    with open(os.path.join(args.output_dir, "model.txt"), "w") as fp:
        fp.write("{}".format(model))

    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=False
    )

    clf_loss = nn.CrossEntropyLoss().cuda()

    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.SGD(
        params_groups, lr=0, momentum=0.9
    )  # we set lr and wd in train_one_epoch

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # cosine lr scheduler with linear warm-up
    args.lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * dist.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    tb_dir = os.path.join(args.output_dir, f"tb-{args.rank}")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir, flush_secs=30)

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(
            model, clf_loss, train_loader, optimizer, epoch, fp16_scaler, args
        )

        # ============ evaluate model ... ============
        test_stats = eval(model, clf_loss, val_loader, epoch, args)

        # ============ saving logs and model checkpoint ... ============
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        if dist.get_rank() == 0:
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
            if args.saveckpt_freq and epoch % args.saveckpt_freq == 0:
                shutil.copy(
                    os.path.join(args.output_dir, "checkpoint.pth"),
                    os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"),
                )

            utils.save_logs(
                train_stats, tb_writer, epoch, (Path(args.output_dir) / "log_train.txt")
            )
            utils.save_logs(
                test_stats, tb_writer, epoch, (Path(args.output_dir) / "log_test.txt")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(model, clf_loss, data_loader, optimizer, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        it = len(data_loader) * epoch + it  # global training iteration

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # move images to gpu
        image = [im.cuda(non_blocking=True) for im in image]

        # we have a list of images as a result of using multi-crop
        # so we repeat labels
        n_crops = args.mc_local_number + args.mc_global_number
        label = label.repeat(n_crops).cuda(non_blocking=True)

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(
                image, image[0], label.chunk(n_crops, dim=0)[0]
            )  # we use only global crops to update the memory
            loss = clf_loss(logit, label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            acc1, acc5 = utils.accuracy(logit.detach(), label, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval(model, clf_loss, data_loader, epoch, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):

        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        output = model(image)

        loss = clf_loss(output, label)
        acc1, acc5 = utils.accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)
    return {f"test/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    args = get_args()
    main(args)
