# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import glob
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:.4f}".format
pd.options.display.max_columns = 300
pd.options.display.width = 1000


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Path to the folder containing the checkpoint file, and all the evaluations.",
)


class Model:
    def __init__(self, root_dir, name=""):
        if name == "":
            name = root_dir.split("/")[-1]
        self.name = name
        self.root_dir = root_dir

    def get_logreg_results(
        self,
        dataset,
        metric_keys=["test/top1", "test/top5"],
    ):

        root_seed_dirs_path = os.path.join(
            self.root_dir,
            dataset,
            "logreg",
            "seed*",
        )
        seed_dirs = sorted(glob.glob(root_seed_dirs_path))
        print(
            "{} logreg seed directories found for model: {}, dataset: {}".format(
                len(seed_dirs), self.name, dataset
            )
        )

        scores = {mk: [] for mk in metric_keys}

        for sdir in seed_dirs:
            log_file = os.path.join(sdir, "logs.pkl")
            if not os.path.isfile(log_file):
                print(
                    "ERROR: No logreg log file found under {}".format(sdir),
                )
                continue

            try:
                log_content = load_pickle(log_file)
            except:
                print("ERROR: Cannot read the log file {}".format(log_file))
                continue

            # we will return top1 and top5 on test set
            for mk in metric_keys:
                score = log_content.get(mk, 0)
                if isinstance(score, list):
                    score = score[-1]
                scores[mk].append(score)

        for mk in metric_keys:
            scores[mk] = np.array(scores[mk])

        return scores


def load_pickle(path):
    with open(path, "rb") as fid:
        return pickle.load(fid)


def get_logreg_results(
    model_list,
    dataset_dict,
    metric_dict={"test/top1": "Top-1"},
):

    results_dict = {
        (dset_title, mt_title): []
        for dset_title in dataset_dict.values()
        for mt_title in metric_dict.values()
    }

    for model in model_list:
        print("-" * 50)
        print(model.name)

        for dset, dset_title in dataset_dict.items():
            results = model.get_logreg_results(
                dset,
                metric_keys=list(metric_dict.keys()),
            )
            for mt, mt_title in metric_dict.items():
                mean = 0.0
                if results is not None:
                    mean = results[mt].mean()
                results_dict[(dset_title, mt_title)].append(mean)

    df = pd.DataFrame(results_dict, index=[model.name for model in model_list])
    return df


def log_odds(acc):
    p = acc / 100
    return np.log(p / (1 - p))


def main(args):
    dataset_dict = OrderedDict()
    dataset_dict["in1k"] = "IN-1K"
    dataset_dict["cog_l1"] = "CoG L-1"
    dataset_dict["cog_l2"] = "CoG L-2"
    dataset_dict["cog_l3"] = "CoG L-3"
    dataset_dict["cog_l4"] = "CoG L-4"
    dataset_dict["cog_l5"] = "CoG L-5"
    dataset_dict["aircraft"] = "Aircraft"
    dataset_dict["cars196"] = "Cars196"
    dataset_dict["dtd"] = "DTD"
    dataset_dict["eurosat"] = "EuroSAT"
    dataset_dict["flowers"] = "Flowers"
    dataset_dict["pets"] = "Pets"
    dataset_dict["food101"] = "Food101"
    dataset_dict["sun397"] = "SUN397"
    dataset_dict["inat2018"] = "iNat-2018"
    dataset_dict["inat2019"] = "iNat-2019"

    #
    model_list = [
        Model(args.model_dir),
        # you can add here more models to compare
    ]

    df = get_logreg_results(
        model_list,
        dataset_dict,
        metric_dict={"test/top1": "Top-1"},
    )
    mean = df.drop(df.columns[0], axis=1).mean(axis=1)
    mean_log_odds = log_odds(df.drop(df.columns[0], axis=1)).mean(axis=1)
    df["Mean"] = mean
    df["Mean Log Odds"] = mean_log_odds
    print(df)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
