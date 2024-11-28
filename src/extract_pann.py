# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import librosa
import lightning as L
import numpy as np
import rootutils
import torch
from gridtk.tools import get_array_job_slice
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
from tqdm import tqdm, trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, manual_repeat_pad, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def load_audio_segment(sample, target_sr):
    vocid = sample[0]
    file = sample[1]
    start = sample[2]
    end = sample[3]
    signal, _ = librosa.load(
        file,
        sr=target_sr,
        mono=True,
        offset=start,
        duration=end - start,
    )
    signal /= np.max(np.abs(signal))
    signal = torch.FloatTensor(signal)
    sig_len = len(signal)
    return vocid, signal, sig_len


def get_batch(samples, start, end, target_sr, min_len):
    wavs, vocids, wavs_len = [], [], []
    end_id = min(end, len(samples))
    for index in range(start, end_id):
        vocid, wav, sig_len = load_audio_segment(samples[index], target_sr)
        wavs.append(wav)
        vocids.append(vocid)
        wavs_len.append(sig_len)

    padded_wavs = collate(wavs, wavs_len, min_len)
    return padded_wavs, wavs_len, vocids


def collate(wavs, wavs_len, min_len):
    max_len = max(min_len, max(wavs_len))
    padded_wavs = torch.vstack([manual_repeat_pad(wav, max_len) for wav in wavs])
    return padded_wavs


def get_features(model, padded_wavs, device):
    with torch.no_grad():
        features = model(padded_wavs.to(device))
    return features


def unpad(hs, hs_len, index):
    hs_unpadded = []
    for h, lens in zip(hs, hs_len):
        l = lens[index]
        hs_unpadded.append(h[index, :l])
    return hs_unpadded


def pack_features(features, vocids, features_dict):
    num_inputs = len(vocids)
    for input_id in range(num_inputs):
        wav_id = str(vocids[input_id])
        # hs = unpad(features, features_len, input_id)
        features_dict[wav_id] = features[input_id].detach().cpu().numpy()
    return features_dict


def extract_pann_feats(samples, target_sr, model, batch_size, device, min_len):
    features_dict = {}
    samples = samples[get_array_job_slice(len(samples))]
    num_wavs = len(samples)

    # Iterate by mini-batches
    for bid in trange(0, len(samples), batch_size):
        end_id = min(bid + batch_size, num_wavs)

        # Get batch
        pwavs, wavs_len, vocids = get_batch(samples, bid, end_id, target_sr, min_len)

        # Get features
        features = get_features(model, pwavs, device)

        # Pack features
        features_dict = pack_features(features, vocids, features_dict)

    # Return
    return features_dict


@task_wrapper
def extract(cfg: DictConfig):
    """Extracts PANN features.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    save_dir = cfg.paths.feats_dir
    target_sr = cfg.sample_rate

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating model at checkpoint ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cfg.paths.model_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    log.info("Checking save dir ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    log.info("Extracting features ...")
    cols = ["vocID", "audio_file", "start", "end"]
    samples = datamodule.data.filelist[cols].values

    stack = extract_pann_feats(
        samples, target_sr, model, cfg.batch_size, device, cfg.min_len
    )

    log.info("Saving features ...")
    # sge_task_id = os.environ.get("SGE_TASK_ID")
    # if sge_task_id is not None or sge_task_id != "undefined":
    #     save_path = os.path.join(save_dir, f"pann_cnn14_{sge_task_id}.npz")
    # else:
    save_path = os.path.join(save_dir, "all_pann_cnn14.npz")
    np.savez(save_path, **stack)

    log.info("Exiting ...")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    _ = extract(cfg)


if __name__ == "__main__":
    main()
