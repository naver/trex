# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import os

import torch
from torchvision.models import resnet50

from . import vision_transformer as vits


def build_model(arch="resnet50", ckpt_file="", ckpt_key="", device="cuda"):

    if arch.startswith("vit"):
        # arch should be in the following form "vit_small_16_cls" or "vit_small_16_gap"
        patch_size = int(arch.split("_")[-2])
        patch_aggregation = arch.split("_")[-1]
        arch = "_".join(arch.split("_")[:2])
        print(
            "==> ViT arch: {}, patch_size: {}, patch_aggregation: {}".format(
                arch, patch_size, patch_aggregation
            )
        )
        model = vits.__dict__[arch](
            patch_size=patch_size, patch_aggregation=patch_aggregation
        )

    else:
        model = resnet50(weights=None)
        model.fc = torch.nn.Identity()

    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, "cpu")
        state_dict = ckpt[ckpt_key] if ckpt_key != "" and ckpt_key in ckpt else ckpt
        state_dict = {
            k.replace("module.", "")
            .replace("model.", "")
            .replace("backbone.", "")
            .replace("encoder.", "")
            .replace("base_encoder.", ""): v
            for k, v in state_dict.items()
        }

        msg = model.load_state_dict(state_dict, strict=False)
        print(" Model checkpoint is loaded with the following message: {}".format(msg))
        assert msg.missing_keys == []
    else:
        print(" No model checkpoint is loaded, using a model with random weights.")

    model = model.to(device)
    model = model.eval()

    return model
