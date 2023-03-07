<!--
trex
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
-->

<!-- omit in toc -->
# Transfer learning evaluations via linear classifiers over pre-extracted features

This folder contains code for running the transfer learning evaluations in our paper [No reason for no supervision: Improving generalization in supervised models](https://arxiv.org/abs/2206.15369), i.e., training linear logistic regression classifiers over pre-extracted features.

- [Installation](#installation)
- [Data](#data)
- [Code](#code)
  - [Extracting features from a pretrained model](#extracting-features-from-a-pretrained-model)
  - [Training logistic regression classifier on pre-extracted features](#training-logistic-regression-classifier-on-pre-extracted-features)
    - [Via L-BFGS implemented in Scikit-learn](#via-l-bfgs-implemented-in-scikit-learn)
    - [Via SGD implemented in PyTorch](#via-sgd-implemented-in-pytorch)
  - [Printing logs](#printing-logs)

# Installation

```bash
conda create -n transfer
conda activate transfer

conda install pytorch=1.13.1 torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

conda install -c intel scikit-learn
# If you have a modern Intel CPU, we recommend installing scikit-learn-intelex
conda install -c intel scikit-learn-intelex

pip install optuna pandas
```

# Data

Using this repo, you can evaluate pretrained models on 16 datasets: ImageNet-1K, 5 ImageNet-CoG levels, 10 smaller-scale datasets.
You can get them as follows:

- ImageNet-1K is available on [the ImageNet website](https://image-net.org/download-images.php).
- i-Naturalist datasets are available on kaggle, requiring an account to download:
[i-Naturalist 2018](https://www.kaggle.com/c/inaturalist-2018),
[i-Naturalist 2019](https://www.kaggle.com/competitions/inaturalist-2019-fgvc6)
- Data for ImageNet-CoG levels can be downloaded from [here](https://github.com/naver/cog).
- To download the remaining 8 datasets, we provide scripts under [scripts/dataset](./scripts/dataset).

# Code

## Extracting features from a pretrained model

We support extracting features from the ResNet50 architecture (defined as in the torchvision package).
You can download a pretrained t-ReX checkpoint from the [main page](../README.md), and use it in the example below for extracting features from that t-ReX model.

To evaluate on other architectures:
- Modify [modeling/builder.py](./modeling/builder.py#9) to build up your architecture, and load your pretrained checkpoint.
There you can see example code for extracting features from ViT architectures.
But we didn't test this functionality thoroughly, so we ask you to use it at your own risk.
- Modify the `<arch>` parameter in [feature_extraction.py](./feature_extraction.py) so that it recognizes your model.

See the example command below for extracting ImageNet-1K features from a pretrained checkpoint.
For other datasets, change the `<dataset>` argument accordingly.
For ImageNet-CoG levels, you also need to provide the paths for the arguments `<cog_levels_mapping_file>` and `<cog_concepts_split_file>` as in [the CoG benchmark](https://github.com/naver/cog).
Also, please see the arguments of [feature_extraction.py](./feature_extraction.py) for other parameters such as batch size (default is 128), number of workers in the dataloader (default is 8), etc.

```bash
ckpt_file=/path/to/checkpoint.pth
# if you downloaded the full checkpoint, set ckpt_key="model"
# otherwise (if you downloaded ResNet50 checkpoint), set it as ckpt_key="none"
ckpt_key=/set/this/as/explained/just/above
dataset="in1k"
dataset_dir=/path/to/imagenet_1k
output_dir=/path/to/save_features
export CUDA_VISIBLE_DEVICES=0  # set this to "" for CPU-only feature extraction

python feature_extraction.py \
    --ckpt_file=${ckpt_file} \
    --ckpt_key=${ckpt_key} \
    --dataset=${dataset} \
    --dataset_dir=${dataset_dir} \
    --output_dir=${output_dir} \
    --cog_levels_mapping_file=/needed/when/extracting/features/of/cog-levels \
    --cog_concepts_split_file=/needed/when/extracting/features/of/cog-levels
```

## Training logistic regression classifier on pre-extracted features

We train linear logistic regression classifiers in two different ways depending on the dataset:
1. Using L-BFGS implemented in Scikit-learn.
  This is useful when the dataset is small (e.g., Aircraft, Cars196, etc.) so that we can learn classifiers quickly without requiring a GPU power.
  We use this approach when training classifiers on the 8 small-scale datasets (see the paper for the full list).
1. Using SGD implemented in PyTorch, supporting GPU-based training.
  This is useful when the dataset is too big (e.g. ImageNet-1K or ImageNet-CoG levels) and CPU-based compute is limited.
  We use this approach when training classifiers on ImageNet-1K, ImageNet-CoG levels and the i-Naturalist datasets.

In both cases, we set their hyper-parameters (i.e., the cost parameter for L-BFGS, and learning rate and weight decay for SGD) via Optuna over 30 trials, see the full list of arguments in [classifier_training.py](./classifier_training.py).
When training a classifier on a dataset, we randomly sample a validation split from its training split.
So, we recommend training classifiers multiple times with different seeds (via the `--seed=${RANDOM}` argument in the examples below) and then taking their average.

Note that [classifier_training.py](./classifier_training.py) can also extract features on-the-fly if needed, please see the full list of arguments.

### Via L-BFGS implemented in Scikit-learn

```bash
features_dir=/<output_dir>/of/the/previous/step
output_dir=/path/to/output/dir/for/the/learned/classifier

# If you don't have scikit-learn-intelex installed, then remove "-m sklearnex"
# We recommend using this package, as it hugely reduces runtime on modern Intel-based CPUs.
python -m sklearnex classifier_training.py \
    --features_dir=${features_dir} \
    --features_norm="none" \
    --clf_type="logreg_sklearn" \
    --output_dir=${output_dir} \
    --seed=${RANDOM}
```

### Via SGD implemented in PyTorch

```bash
features_dir=/path/to/pre-extracted/features/from/the/previous/step
output_dir=/path/to/output/dir/for/the/learned/classifier
export CUDA_VISIBLE_DEVICES=0

# L2 normalization follows the convention in the ImageNet-CoG benchmark.
python classifier_training.py \
    --features_dir=${features_dir} \
    --features_norm="l2" \
    --clf_type="logreg_torch" \
    --output_dir=${output_dir} \
    --seed=${RANDOM}
```

## Printing logs

You can use [print_results.py](./scripts/print_results.py) to print the results on all datasets as follows:

```bash
python scripts/print_results.py --model_dir=/path/to/directory/containing/checkpoint.pth
```
