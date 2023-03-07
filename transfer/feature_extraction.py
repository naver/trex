# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import os
import sys
from time import time

import data
import modeling
import torch
from torch.utils.data import DataLoader

import utils


def main(args):
    """
    Main routine to extract features.
    """

    print("==> Initializing the pretrained model")
    model = modeling.build_model(
        args.arch, ckpt_file=args.ckpt_file, ckpt_key=args.ckpt_key, device=args.device
    )

    # extract features from training and test sets
    for split in ("trainval", "test"):
        dataset = data.load_dataset(
            args.dataset,
            args.dataset_dir,
            split,
            args.dataset_image_size,
            cog_levels_mapping_file=args.cog_levels_mapping_file,
            cog_concepts_split_file=args.cog_concepts_split_file,
        )
        print(
            "==> Extracting features from {} / {} (size: {})".format(
                args.dataset, split, len(dataset)
            )
        )
        print(" Data loading pipeline: {}".format(dataset.transform))
        X, Y = extract_features_loop(
            model, dataset, args.batch_size, args.n_workers, args.device
        )

        features_file = os.path.join(args.output_dir, "features_{}.pth".format(split))
        print(" Saving features under {}".format(features_file))
        torch.save({"X": X, "Y": Y}, features_file)


def extract_features_loop(
    model, dataset, batch_size=128, n_workers=12, device="cuda", print_iter=50
):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
    )

    # (feature, label) pairs to be stored under args.output_dir
    X = None
    Y = None
    six = 0  # sample index

    t_per_sample = utils.AverageMeter("time-per-sample")
    t0 = time()

    with torch.no_grad():
        for bix, batch in enumerate(dataloader):
            assert (
                len(batch) == 2
            ), "Data loader should return a tuple of (image, label) every iteration."
            image, label = batch
            feature = model(image.to(device))

            if X is None:
                print(
                    " Size of the first batch: {} and features {}".format(
                        list(image.shape), list(feature.shape)
                    ),
                    flush=True,
                )
                X = torch.zeros(
                    len(dataset), feature.size(1), dtype=torch.float32, device="cpu"
                )
                Y = torch.zeros(len(dataset), dtype=torch.long, device="cpu")

            bs = feature.size(0)
            X[six : six + bs] = feature.cpu()
            Y[six : six + bs] = label
            six += bs

            t1 = time()
            td = t1 - t0
            t_per_sample.update(td / bs, bs)
            t0 = t1

            if (bix % print_iter == 0) or (bix == len(dataloader) - 1):
                print(
                    " {:6d}/{:6d} extracted, {:5.3f} secs per sample, {:5.1f} mins remaining".format(
                        six,
                        len(dataset),
                        t_per_sample.avg,
                        (t_per_sample.avg / 60) * (len(dataset) - six),
                    ),
                    flush=True,
                )

    assert six == len(X)
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50"],
        help="The architecture of the pretrained model",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        required=True,
        help="Model checkpoint file",
    )
    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="",
        help="""Key in the checkpoint dictionary that corresponds to the model's state_dict
        For instance, if the checkpoint dictionary contains
        {'optimizer': optimizer.state_dict(),
         'model': model.state_dict(),
         ...}
        then this argument should be 'model'.
        If the checkpoint dictionary is the model state_dict itself, leave this argument empty.""",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="in1k",
        choices=[
            "in1k",
            "cog_l1",
            "cog_l2",
            "cog_l3",
            "cog_l4",
            "cog_l5",
            "aircraft",
            "cars196",
            "dtd",
            "eurosat",
            "flowers",
            "pets",
            "food101",
            "sun397",
            "inat2018",
            "inat2019",
        ],
        help="From which datasets to extract features",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset_image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to extract features.",
    )
    parser.add_argument(
        "--cog_levels_mapping_file",
        type=str,
        help="Pickle file containing a list of concepts in each level (5 lists in total)."
        'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--cog_concepts_split_file",
        type=str,
        help="Pickle file containing training and test splits for each concept in ImageNet-CoG."
        'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_file):
        print(
            "Checkpoint file ({}) not found. "
            "Please provide a valid checkpoint file path for the pretrained model.".format(
                args.ckpt_file
            )
        )
        sys.exit(-1)

    if not os.path.isdir(args.dataset_dir):
        print(
            "Dataset not found under {}. "
            "Please provide a valid dataset path".format(args.dataset_dir)
        )
        sys.exit(-1)

    if args.dataset in ["cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5"] and not (
        os.path.isfile(args.cog_levels_mapping_file)
        and os.path.isfile(args.cog_concepts_split_file)
    ):
        print(
            "ImageNet-CoG files are not found. "
            "Please check the <cog_levels_mapping_file> and <cog_concepts_split_file> arguments."
        )
        sys.exit(-1)

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
        print("No CUDA-compatible device found. " "We will only use CPU.")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    utils.print_program_info(args)

    main(args)
