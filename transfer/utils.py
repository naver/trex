# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import os
import pickle
import shutil
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_program_info(args, save_path=None):
    """
    Prints argparse arguments, and saves them into a file if save_path is provided.
    """

    fid = None
    if save_path is not None:
        fid = open(save_path, "w")

    def _print(text):
        print(text)
        if fid is not None:
            print(text, file=fid, flush=True)

    _print("-" * 100)
    _print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    _print("-" * 100)
    _print("MKL_NUM_THREADS={}".format(os.environ.get("MKL_NUM_THREADS", "(unset)")))
    _print("OMP_NUM_THREADS={}".format(os.environ.get("OMP_NUM_THREADS", "(unset)")))
    _print(sys.argv[0])
    for parg in sys.argv[1:]:
        _print("\t{}".format(parg))
    _print("-" * 100)

    if fid is not None:
        fid.flush()
        fid.close()


def save_pickle(obj, save_path):
    with open(save_path, "wb") as fid:
        pickle.dump(obj, fid)


def load_pickle(save_path):
    with open(save_path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def preprocess_features(
    train_features,
    test_features,
    args,
):

    if args.features_norm == "standard":
        _mean = train_features.mean(dim=0)
        _std = train_features.std(dim=0)

        train_features = (train_features - _mean) / _std.clip(1e-5)
        test_features = (test_features - _mean) / _std.clip(1e-5)

    elif args.features_norm == "l2":
        train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
        test_features = torch.nn.functional.normalize(test_features, p=2, dim=1)

    return [
        train_features.detach().clone(),
        test_features.detach().clone(),
    ]


def print_feature_info(split, X, Y):
    print(
        "==> Split: {:8s} | features.shape:{}, features.norm:{:.3f}, labels.shape:{}, labels.n_unique:{}".format(
            split,
            list(X.shape),
            X.norm(dim=1).mean(),
            list(Y.shape),
            len(torch.unique(Y)),
        )
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=2)
