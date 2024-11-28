# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import seaborn as sns
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from scipy.spatial.distance import cdist, pdist
from torch.utils.data import DataLoader
import itertools

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    collate_fn,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def pdists(cfg: DictConfig):

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if "layer_id" in cfg:
        layer_id = cfg.layer_id
    else:
        layer_id = ""

    calltypes_to_index = {
        "Peep(Pre-Phee)": 0,
        "Phee": 1,
        "Twitter": 2,
        "Trill": 3,
        "Trillphee": 4,
        "Tsik Tse": 5,
        "Egg": 6,
        "Pheecry(cry)": 7,
        "TrllTwitter": 8,
        "Pheetwitter": 9,
        "Peep": 10,
    }
    # Get datamodules
    log.info(f"Instantiating datamodules <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    savedir = cfg.paths.pkl_dir

    # Get train sets
    log.info("Setting up datamodules")
    datamodule.setup()
    train_dataset = datamodule.train_dataset

    # Pass entire dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        collate_fn=collate_fn,
    )

    x, y, _ = next(iter(train_dataloader))

    if cfg.mname == "wavlm":  # only take the means
        x = x[:, :, : x.shape[-1] // 2]

    # # Old code for intra-class only
    # # Iterate over calltypes
    # if cfg.data.data.selected_labels == "calltype":
    #     for k, v in calltypes_to_index.items():
    #         log.info(f"{v}")
    #         call = x[y == v].squeeze(1)
    #         distances = pdist(call, "cosine")
    #         os.makedirs(savedir, exist_ok=True)
    #         np.savez(os.path.join(savedir, f"{cfg.mname}_{v}_pdists.npz"), distances)

    # elif cfg.data.data.selected_labels == "caller":
    #     for clid in range(0, 10):
    #         log.info(f"{clid}")
    #         call = x[y == clid].squeeze(1)
    #         distances = pdist(call, "cosine")
    #         os.makedirs(savedir, exist_ok=True)
    #         np.savez(os.path.join(savedir, f"{cfg.mname}_{clid}_pdists.npz"), distances)

    # Iterate over all combinations of call types, including same-type (intra-class) combinations
    if cfg.data.data.selected_labels == "calltype":
        for (type1, id1), (type2, id2) in itertools.combinations_with_replacement(
            calltypes_to_index.items(), 2
        ):
            call1 = x[y == id1].squeeze(1)
            call2 = x[y == id2].squeeze(1)

            log.info(id1, id2)

            # Use cdist for inter-class distances and pdist for intra-class distances
            if id1 == id2:
                distances = pdist(call1, "cosine")
            else:
                distances = cdist(call1, call2, "cosine")

            # Save the distances with appropriate naming
            os.makedirs(savedir, exist_ok=True)
            filename = f"{cfg.mname}_{id1}_vs_{id2}_distances.npz"
            np.savez_compressed(os.path.join(savedir, filename), distances)

    elif cfg.data.data.selected_labels == "caller":
        for id1, id2 in itertools.combinations_with_replacement(range(0, 10), 2):
            call1 = x[y == id1].squeeze(1)
            call2 = x[y == id2].squeeze(1)

            log.info(id1, id2)

            # Use cdist for inter-class distances and pdist for intra-class distances
            if id1 == id2:
                distances = pdist(call1, "cosine")
            else:
                distances = cdist(call1, call2, "cosine")

            # Save the distances with appropriate naming
            os.makedirs(savedir, exist_ok=True)
            filename = f"{cfg.mname}_{id1}_vs_{id2}_distances.npz"
            np.savez_compressed(os.path.join(savedir, filename), distances)

    log.info("Finished !")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="pdist.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # train the model
    _ = pdists(cfg)


if __name__ == "__main__":
    main()
