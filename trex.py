# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models

from multi_crop import MultiCropWrapper
from utils import concat_all_gather


class tReX(nn.Module):
    def __init__(
        self,
        arch="resnet50",
        proj_input_l2_norm=True,
        proj_hidden_layers=3,
        proj_hidden_dim=2048,
        proj_bottleneck_dim=256,
        clf_tau=0.1,
        memory_size=8192,
        n_classes=1000,
        ema_momentum=0.999,
    ):
        super().__init__()

        ##########
        # Model that we train
        self.model = make_model(
            arch,
            proj_input_l2_norm,
            proj_hidden_layers,
            proj_hidden_dim,
            proj_bottleneck_dim,
        )
        ##########

        self.memory = None
        self.clf = None
        self.norm = None

        if memory_size > 0:
            # to normalize projector outputs
            self.norm = L2Norm(dim=1)

            ##########
            # Memory bank
            self.clf_tau = clf_tau
            self.memory = Memory(proj_bottleneck_dim, memory_size, n_classes)
            ##########

            ##########
            # Model EMA for memory
            self.ema_momentum = ema_momentum
            self.model_ema = make_model(
                arch,
                proj_input_l2_norm,
                proj_hidden_layers,
                proj_hidden_dim,
                proj_bottleneck_dim,
            )
            for param_m, param_ema in zip(
                self.model.parameters(), self.model_ema.parameters()
            ):
                param_ema.data.copy_(param_m.data)
                param_ema.requires_grad = False
            ##########

        else:
            self.clf = ClfLayer(proj_bottleneck_dim, n_classes, tau=clf_tau)

    def forward(self, image, image_memory=None, label_memory=None):
        x = self.model(image)

        # compute class logits
        if self.clf is not None:
            # using learnable class weights
            return self.clf(x)
        else:
            # using class prototypes from the memory
            w = self.memory.class_weights()  # already l2 normalized
            x = self.norm(x)
            logits = (x @ w.T) / self.clf_tau

            # update the memory
            if image_memory is not None and label_memory is not None:
                with torch.no_grad():
                    self.update_model_ema()
                    x_ema = self.model_ema(image_memory)
                    self.memory.update(x_ema, label_memory)
            return logits

    @torch.no_grad()
    def update_model_ema(self):
        for param_m, param_ema in zip(
            self.model.parameters(), self.model_ema.parameters()
        ):
            param_ema.data = param_ema.data * self.ema_momentum + param_m.data * (
                1.0 - self.ema_momentum
            )


class Projector(nn.Module):
    def __init__(
        self,
        ft_dim,
        input_l2_norm=True,
        hidden_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()

        # list of MLP layers
        layers = []

        if input_l2_norm:
            layers.append(L2Norm(dim=1))

        # hidden layers
        _in_dim = ft_dim
        for _ in range(hidden_layers):
            layers.append(MLPLayer(_in_dim, hidden_dim))
            _in_dim = hidden_dim

        # bottleneck layer
        layers.append(nn.Linear(_in_dim, bottleneck_dim, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ClfLayer(nn.Module):
    def __init__(self, emb_dim, n_classes=1000, tau=0.1):
        super().__init__()
        self.tau = tau

        self.norm = nn.Identity()
        if tau > 0:
            self.norm = L2Norm(dim=1)

        self.fc = nn.Linear(emb_dim, n_classes, bias=False)

    def forward(self, x):
        # no temperature scaling
        if self.tau <= 0:
            return self.fc(x)

        # temperature scaling with l2-normalized weights
        x = self.norm(x)
        w = self.norm(self.fc.weight)
        o = (x @ w.t()) / self.tau
        return o

    def extra_repr(self):
        return "tau={}".format(self.tau)


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return nn.functional.normalize(x, dim=self.dim, p=2)


class Memory(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_size=65536,
        n_classes=1000,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.n_classes = n_classes
        self.norm = L2Norm(dim=1)

        self.register_buffer(
            "features",
            torch.randn(memory_size, feature_dim),
        )
        self.register_buffer("labels", torch.randint(0, n_classes, (memory_size,)))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def extra_repr(self) -> str:
        return (
            "feature_dim={}, "
            "memory_size={}, "
            "n_classes={}".format(
                self.feature_dim,
                self.memory_size,
                self.n_classes,
            )
        )

    @torch.no_grad()
    def class_weights(self) -> torch.Tensor:
        """
        Returns l2-normalized class weights by averaging the features from the same class
        """
        weights = torch.zeros(self.n_classes, self.feature_dim).to(self.features)
        weights.index_add_(0, self.labels, self.features)
        return self.norm(weights)

    @torch.no_grad()
    def update(self, features, labels):
        """
        Updates either the memory bank or moving class means.
        """
        # gather the features first
        features = concat_all_gather(features)
        labels = concat_all_gather(labels)
        assert features.shape[0] == labels.shape[0]
        batch_size, feature_dim = features.shape

        # enqueue features and labels to the memory
        # if there is not enough space in the memory
        # discard some of the them
        ptr = int(self.ptr)
        if ptr + batch_size > self.memory_size:
            features = features[: self.memory_size - ptr]
            labels = labels[: self.memory_size - ptr]
            batch_size = features.shape[0]

        # replace the features and labels at ptr
        self.features[ptr : ptr + batch_size, :] = features
        self.labels[ptr : ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.memory_size  # move the pointer
        self.ptr[0] = ptr


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def make_model(
    arch, proj_input_l2_norm, proj_hidden_layers, proj_hidden_dim, proj_bottleneck_dim
):
    ft_dim_dict = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }
    assert (
        arch in ft_dim_dict.keys()
    ), "Error: Unsupported architecture: {} (supported ones: {})".format(
        arch, ft_dim_dict.keys()
    )

    encoder = models.__dict__[arch]()
    head = Projector(
        ft_dim_dict[arch],
        input_l2_norm=proj_input_l2_norm,
        hidden_layers=proj_hidden_layers,
        hidden_dim=proj_hidden_dim,
        bottleneck_dim=proj_bottleneck_dim,
    )
    head.apply(init_weights)
    model = MultiCropWrapper(encoder, head)
    return model
