##################################################
# Multi-crop related code re-used from DINO
# https://github.com/facebookresearch/dino
##################################################

import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps


class MultiCropAugmentation(object):
    def __init__(self, global_number, global_scale, local_number, local_scale):
        assert (global_number > 0) or (local_number > 0)
        self.global_number = global_number
        self.local_number = local_number

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_tfm = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=global_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(0.2),
                Solarization(0.2),
                normalize,
            ]
        )

        # transformation for the local small crops
        self.local_tfm = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96,
                    scale=local_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __repr__(self) -> str:
        return (
            "global_number={}, local_number={}, \nglobal_tfm={}\nlocal_tfm={}".format(
                self.global_number, self.local_number, self.global_tfm, self.local_tfm
            )
        )

    def __call__(self, image):
        crops = []

        for _ in range(self.global_number):
            crops.append(self.global_tfm(image))

        for _ in range(self.local_number):
            crops.append(self.local_tfm(image))

        return crops


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, encoder, head):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        encoder.fc, encoder.head = nn.Identity(), nn.Identity()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.encoder(torch.cat(x[start_idx:end_idx]))
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features
        return self.head(output)
