<!--
trex
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
-->

<!-- omit in toc -->
# No reason for no supervision: Improved generalization in supervised models

| [Project Website](https://europe.naverlabs.com/t-rex) | [Paper (arXiv)](https://arxiv.org/abs/2206.15369) | [Paper (ICLR 2023 - notable top 25%)](https://openreview.net/forum?id=3Y5Uhf5KgGK) |
| :-: | :-: | :-: |


In this repository, we provide:
- Several pretrained t-ReX and t-ReX* models in PyTorch (see [here](#model-zoo)).
- Code for training our t-ReX and t-ReX* models on the ImageNet-1K dataset in PyTorch (see [here](#training-t-rex-models)).
- Code for running transfer learning evaluations of pretrained models via linear classification over pre-extracted features on 16 downstream datasets (see [here](#transfer-learning-evaluation-suite)).

If you find this repository useful, please consider citing us:
```
@inproceedings{sariyildiz2023improving,
    title={No Reason for No Supervision: Improved Generalization in Supervised Models},
    author={Sariyildiz, Mert Bulent and Kalantidis, Yannis and Alahari, Karteek and Larlus, Diane},
    booktitle={International Conference on Learning Representations},
    year={2023},
}
```

- [Model Zoo](#model-zoo)
- [Training t-ReX models](#training-t-rex-models)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training commands](#training-commands)
    - [Commands for t-ReX-OCM models](#commands-for-t-rex-ocm-models)
    - [Commands for plain t-ReX models](#commands-for-plain-t-rex-models)
- [Transfer learning evaluation suite](#transfer-learning-evaluation-suite)


# Model Zoo

In the table below, we provide links for several pretrained t-ReX and t-ReX* models.
These are the models which produce the results reported in the paper, as well as the models reproduced with the cleaner codebase released in this repo.
Transfer performance of these models are averaged over 15 datasets, which include two additions, i.e., the i-Naturalist datasets, to the 13 transfer datasets we mainly used in the paper.
To perform transfer evaluations, see the [corresponding section of this readme](#evaluating-t-rex-models-on-transfer-datasets).

<table style="border: 1px;">
    <tr>
        <th>Model</th>
        <th style="text-align: center">ResNet50 <br/> Checkpoint</th>
        <th style="text-align: center">Full <br/> Checkpoint</th>
        <th style="text-align: center">ImageNet-1K <br/> (Top-1 %)</th>
        <th style="text-align: center">Average Transfer <br/> (Log odds)</th>
    </tr>
    <tr>
        <td colspan="5"><i>Models reported in the paper</i></td>
    </tr>
    <tr>
        <td> t-ReX </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trex.pth">Link</a> </td>
        <td style="text-align: center"></td>
        <td style="text-align: center">78.0</td>
        <td style="text-align: center">1.1704</td>
    </tr>
    <tr>
        <td> t-ReX* </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trexstar.pth">Link</a> </td>
        <td style="text-align: center"></td>
        <td style="text-align: center">80.2</td>
        <td style="text-align: center">0.8829</td>
    </tr>
    <tr>
        <td colspan="5"><i>Models reproduced with this code base</i></td>
    </tr>
    <tr>
        <td> t-ReX </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trex_2.pth">Link</a> </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trex_2_checkpoint_full.pth">Link</a> </td>
        <td style="text-align: center">77.9</td>
        <td style="text-align: center">1.1664</td>
    </tr>
    <tr>
        <td> t-ReX* </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trexstar_2.pth">Link</a> </td>
        <td style="text-align: center"> <a href="https://download.europe.naverlabs.com/ComputerVision/trex_models/trexstar_2_checkpoint_full.pth">Link</a> </td>
        <td style="text-align: center">80.2</td>
        <td style="text-align: center">0.8800</td>
    </tr>
</table>

Full checkpoints contain a separate state dictionary for the model, optimizer and gradient scaler (for mixed precision).
We share them for reference.
Whereas, you can use the ResNet50 checkpoints simply by

```python
import torch as th
from torchvision.models import resnet50
ckpt = th.load("trex.pth", "cpu")
net = resnet50()
msg = net.load_state_dict(ckpt, strict=False)
assert msg.missing_keys == ["fc.weight", "fc.bias"] and msg.unexpected_keys == []
```

# Training t-ReX models

## Installation

We developed this code by using a recent version of PyTorch, torchvision and Tensorboard.
We recommend creating a new conda environment to manage these packages.
```bash
conda create -n trex
conda activate trex
conda install pytorch=1.13.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tensorboard
```

## Dataset

We train our models on the ILSVRC-2012 dataset (also called ImageNet-1K).
It is available on [the ImageNet website](https://image-net.org/download-images.php).
Once you download the dataset, make sure that `data_dir=/path/to/imagenet` contains `train` and `val` directories, each including 1000 sub-directories for the images of the ImageNet-1K classes.

## Training commands

Below, we provide commands for training plain t-ReX and t-ReX-OCM models on ImageNet-1K.
Note that the results we report in the paper are obtained by 100 epoch trainings over 4 GPUs each processing a batch of 64 samples.
If you want to use a less number of GPUs or increase the batch size, etc., see the arguments of [main.py](./main.py).

### Commands for t-ReX-OCM models

t-ReX-OCM models are defined by Equation-2 of the paper.

<details>
<summary> Command for training a t-ReX-OCM-1 model (named <strong>t-ReX*</strong> in the paper)</summary>

```bash
data_dir=/path/to/imagenet
output_dir=/path/where/to/save/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3  # change accordingly the <nproc_per_node> argument below

python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=1 \
    --mc_global_scale 0.40 1.00 \
    --mc_local_scale 0.05 0.40
```

</details>

<details>
<summary> Command for training a t-ReX-OCM-3 model (named <strong>t-ReX</strong> in the paper)</summary>

```bash
data_dir=/path/to/imagenet
output_dir=/path/where/to/save/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3  # change accordingly the <nproc_per_node> argument below

python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=3 \
    --mc_global_scale 0.25 1.00 \
    --mc_local_scale 0.05 0.25
```

</details>

### Commands for plain t-ReX models

Plain t-ReX models are defined by Equation-1 of the paper.
Compared to the commands for training OCM models above, we just add the ```--memory_size=0``` argument, which disables the OCM part.

<details>
<summary> Command for training a plain t-ReX-1 model</summary>

```bash
data_dir=/path/to/imagenet
output_dir=/path/where/to/save/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3  # change accordingly the <nproc_per_node> argument below

python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=1 \
    --mc_global_scale 0.40 1.00 \
    --mc_local_scale 0.05 0.40 \
    --memory_size=0
```

</details>

<details>
<summary> Command for training a plain t-ReX-3 model</summary>

```bash
data_dir=/path/to/imagenet
output_dir=/path/where/to/save/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3  # change accordingly the <nproc_per_node> argument below

python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py  \
    --output_dir=${output_dir} \
    --data_dir=${data_dir} \
    --seed=${RANDOM} \
    --pr_hidden_layers=3 \
    --mc_global_scale 0.25 1.00 \
    --mc_local_scale 0.05 0.25 \
    --memory_size=0
```

</details>


# Transfer learning evaluation suite

We provide the evaluation code under the [transfer](./transfer) folder.
Please navigate there.


<!-- omit in toc -->
# Acknowledgement
Our implementation builds on several public code repositories such as [DINO](https://github.com/facebookresearch/dino), [MoCo](https://github.com/facebookresearch/moco) and [the PyTorch examples](https://github.com/pytorch/examples).
We thank all the authors and developers for making their code accessible.
