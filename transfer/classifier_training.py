# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import os
import random

import data
import modeling
import numpy as np
import optuna
import torch
from feature_extraction import extract_features_loop
from logreg_trainer import LogregSklearnTrainer, LogregTorchTrainer

import utils


def _prepare_features(args):

    # load or extract features from the trainval and test splits
    model = None
    data_dict = {}

    for split in ("trainval", "test"):

        features_file = os.path.join(args.features_dir, "features_{}.pth".format(split))
        if os.path.isfile(features_file):
            print("==> Loading pre-extracted features from {}".format(features_file))
            features_dict = torch.load(features_file, "cpu")
            X, Y = features_dict["X"], features_dict["Y"]

        else:
            print(
                "==> No pre-extracted features found, extracting them under {}".format(
                    features_file
                )
            )

            if model is None:
                print("==> Initializing model")
                model = modeling.build_model(
                    args.arch,
                    ckpt_file=args.ckpt_file,
                    ckpt_key=args.ckpt_key,
                    device=args.device,
                )

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
                model,
                dataset,
                batch_size=args.ftext_batch_size,
                n_workers=args.ftext_n_workers,
                device=args.device,
            )

            print(" Saving features under {}".format(features_file))
            torch.save({"X": X, "Y": Y}, features_file)

        data_dict[split] = [X, Y]

    # split the trainval into train and val
    data_dict["train"], data_dict["val"] = data.utils.split_trainval(
        data_dict["trainval"][0], data_dict["trainval"][1], per_val=args.dataset_per_val
    )

    return data_dict


def main(args):

    data_dict = _prepare_features(args)

    trainer_class = (
        LogregSklearnTrainer
        if args.clf_type == "logreg_sklearn"
        else LogregTorchTrainer
    )

    # tune hyper-parameters with optuna
    print("==> Starting hyper-parameter tuning")
    clf_trainer = trainer_class(
        data_dict["train"][0],
        data_dict["train"][1],
        data_dict["val"][0],
        data_dict["val"][1],
        args,
    )
    hps_sampler = optuna.samplers.TPESampler(
        multivariate=args.clf_type == "logreg_torch",
        group=args.clf_type == "logreg_torch",
        seed=args.seed,
    )
    study = optuna.create_study(sampler=hps_sampler, direction="maximize")
    study.optimize(
        clf_trainer,
        n_trials=args.n_optuna_trials,
        n_jobs=args.n_optuna_workers,
        show_progress_bar=False,
    )
    utils.save_pickle(study, os.path.join(args.output_dir, "study.pkl"))
    fig = optuna.visualization.plot_contour(study, params=clf_trainer.hps_list)
    fig.write_html(os.path.join(args.output_dir, "study_contour.html"))

    print("*" * 50)
    print("Hyper-parameter search ended")
    print("best_trial:")
    print(str(study.best_trial))
    print("best_params:")
    print(str(study.best_params))
    print("*" * 50, flush=True)

    # train the final classifier with the tuned hyper-parameters
    print("==> Training the final classifier")
    del clf_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clf_trainer = trainer_class(
        data_dict["trainval"][0],
        data_dict["trainval"][1],
        data_dict["test"][0],
        data_dict["test"][1],
        args,
    )
    clf_trainer.set_hps(study.best_params)
    clf_trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        type=str,
        default="",
        help="Directory to include features_trainval.pth and features_test.pth",
    )
    parser.add_argument(
        "--features_norm",
        type=str,
        default="none",
        choices=["standard", "l2", "none"],
        help="Normalization applied to features before the classifier",
    )
    parser.add_argument(
        "--clf_type",
        type=str,
        default="logreg_sklearn",
        choices=["logreg_sklearn", "logreg_torch"],
        help="Type of linear classifier to train on top of features",
    )
    parser.add_argument(
        "--dataset_per_val",
        type=float,
        default=0.2,
        help="Percentage of the val set, sampled from the trainval set for hyper-parameter tuning",
    )
    # For the L-BFGS-based logistic regression trainer implemented in scikit-learn
    parser.add_argument(
        "--clf_C",
        type=float,
        help="""Inverse regularization strength for sklearn.linear_model.LogisticRegression.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_C_min",
        type=float,
        default=1e-5,
        help="Power of the minimum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_C_max",
        type=float,
        default=1e6,
        help="Power of the maximum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_max_iter",
        type=int,
        default=2000,
        help="Maximum number of iterations to run the classifier for sklearn.linear_model.LogisticRegression during the hyper-parameter tuning stage.",
    )
    # For the SGD-based logistic regression trainer implemented in PyTorch
    parser.add_argument(
        "--clf_lr",
        type=float,
        help="""Learning rate.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_lr_min",
        type=float,
        default=1e-1,
        help="Power of the minimum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_lr_max",
        type=float,
        default=1e2,
        help="Power of the maximum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd",
        type=float,
        help="""Weight decay.
        Note that this variable is determined by Optuna, do not set it manually""",
    )
    parser.add_argument(
        "--clf_wd_min",
        type=float,
        default=1e-12,
        help="Power of the minimum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd_max",
        type=float,
        default=1e-4,
        help="Power of the maximum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_mom",
        type=float,
        default=0.9,
        help="SGD momentum. We do not tune this variable.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=100,
        help="""Number of epochs to train the linear classifier.
        We do not tune this variable""",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=1024,
        help="""Batch size for SGD.
        We do not tune this variable""",
    )
    # Common for all trainers
    parser.add_argument(
        "--n_sklearn_workers",
        type=int,
        default=-1,
        help="Number of CPU cores to use in Scikit-learn jobs. -1 means to use all available cores.",
    )
    parser.add_argument(
        "--n_optuna_workers",
        type=int,
        default=1,
        help="Number of concurrent Optuna jobs",
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=30,
        help="Number of trials run by Optuna",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generators",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction and classifier training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./linear-classifier-output",
        help="Whether to save the logs",
    )
    # The following arguments are needed if features have not been extracted yet.
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet50"],
        help="""The architecture of the pre-trained model""",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="",
        help="Model checkpoint file",
    )
    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="",
        help="Key in the checkpoint dictionary that corresponds to the model state_dict",
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
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset_image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
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
        "--ftext_batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--ftext_n_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )

    args = parser.parse_args()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"
    utils.print_program_info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
