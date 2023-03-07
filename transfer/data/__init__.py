# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import sys

from torchvision import transforms

from . import (aircraft, cars196, dtd, eurosat, flowers, food101, imagenet_cog,
               in1k, inat, pets, sun397, utils)


def load_dataset(
    dataset,
    dataset_dir,
    split,
    image_size=224,
    cog_levels_mapping_file="",
    cog_concepts_split_file="",
) -> utils.TransferDataset:
    """
    Loads a split of a dataset with the center-crop augmentation.
    """

    assert split in ("trainval", "test"), "Unrecognized split: {}".format(split)

    transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if dataset.startswith("cog_"):
        return imagenet_cog.load_split(
            dataset,
            dataset_dir,
            split,
            transform,
            cog_levels_mapping_file,
            cog_concepts_split_file,
        )
    elif dataset.startswith("inat"):
        year = dataset.replace("inat", "")
        return inat.load_split(dataset_dir, year, split, transform)
    else:
        return (
            sys.modules[__name__]
            .__dict__[dataset]
            .load_split(dataset_dir, split, transform)
        )
