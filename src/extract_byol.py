# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os
from typing import Any, Optional

import hydra
import librosa
import lightning as L
import numpy as np
import rootutils
import torch
from gridtk.tools import get_array_job_slice
from lightning import LightningDataModule
from omegaconf import DictConfig
from s3prl.nn import S3PRLUpstream
from s3prl.util.download import set_dir
from tqdm import tqdm, trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def get_model(upstream_model, device="cuda"):
    return S3PRLUpstream(upstream_model).to(device).eval()


def unpad(hs, hs_len, index):
    hs_unpadded = []
    for h, lens in zip(hs, hs_len):
        l = lens[index]
        hs_unpadded.append(h[index, :l])
    return hs_unpadded

def collate(wavs, padding_value: int = 0):
    from torch.nn.utils.rnn import pad_sequence

    padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=padding_value)
    return padded_wavs


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
    sig_len = len(signal)
    return vocid, torch.FloatTensor(signal), sig_len


def get_batch(samples, start, end, target_sr):
    wavs, vocids, wavs_len = [], [], []
    end_id = min(end, len(samples))
    for index in range(start, end_id):
        vocid, wav, sig_len = load_audio_segment(samples[index], target_sr)
        wavs.append(wav)
        vocids.append(vocid)
        wavs_len.append(sig_len)

    padded_wavs = collate(wavs)
    return padded_wavs, wavs_len, vocids


def get_features(model, padded_wavs, wavs_len, device="cuda"):
    with torch.no_grad():
        wavs_len = torch.LongTensor(wavs_len).to(device)
        all_hs, all_hs_len = model(padded_wavs.to(device), wavs_len)
    return all_hs[0], all_hs_len


def compute_functionals(features_dict, functionals_dict):
    """Computes functionals from the features."""
    # Iterate
    for vocid, embed in features_dict.items():  # Iterate over vocids
        # Iterate over layers
        layer_list = []
        for i in range(len(embed)):
            # data[vocid][l].shape is (num_frames, embedding_length)
            # We average across the num_frames, so the length will be (1, embedding_length)
            # We average for mean and std, and then concatenate both vectors to get a final
            # functional size of (1, 2 * embedding_length)
            means = torch.mean(embed[i], dim=0)
            stds = torch.std(embed[i], dim=0)
            joint_functional = torch.cat((means, stds), dim=0)

            # Append to list, which will contains all layers for this vocid utterance
            layer_list.append(joint_functional.detach().cpu().numpy())

        # Save list with all layers to dict with vocid as key
        functionals_dict[str(vocid)] = np.vstack(layer_list)

    return functionals_dict


def pack_features(hidden_states, hidden_states_len, ids, features_dict):
    num_inputs = len(ids)
    for input_id in range(num_inputs):
        wav_id = ids[input_id]
        #hs = unpad(hidden_states, hidden_states_len, input_id)
        features_dict[str(wav_id)] = hidden_states[input_id].detach().cpu().numpy()
    return features_dict


def average_embeddings(feat):
    """
    BYOL returns framewise embeddings if input_len > 16000: [B, N, 2048]
    We average them to get a single scene embedding: [B, 1, 2048]
    """
    return feat.mean(dim=1, keepdim=True)


def extract_feats(samples, target_sr, model, batch_size, device="cuda"):
    features_dict = {}
    # stack = {}
    samples = samples[get_array_job_slice(len(samples))]
    num_wavs = len(samples)

    # Iterate by mini-batches
    for bid in trange(0, len(samples), batch_size):
        end_id = min(bid + batch_size, num_wavs)

        # Get batch
        pwavs, wavs_len, vocids = get_batch(samples, bid, end_id, target_sr)

        # Get features
        all_hs, all_hs_len = get_features(model, pwavs, wavs_len, device)

        if all_hs.shape[1] != 1:
            all_hs = average_embeddings(all_hs)

        # Pack features
        features_dict = pack_features(all_hs, all_hs_len, vocids, features_dict)

        # Compute functionals
        #stack = compute_functionals(features_dict, stack)

    # Return
    return features_dict


@task_wrapper
def extract(cfg: DictConfig):
    """Extracts features

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    save_dir = cfg.paths.feats_dir
    target_sr = cfg.sample_rate
    batch_size = cfg.batch_size

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Instantiating model <{cfg.model}>")
    try:
        set_dir(cfg.paths.s3prl_cache_path)
    except Exception as e:
        print(f"Unable to set the path to -> {cfg.paths.s3prl_cache_path}")

    model = get_model(cfg.model, device)

    log.info("Checking save dir ...")
    if not os.path.exists(save_dir):  # If folder doesn't exist
        os.makedirs(save_dir, exist_ok=False)  # Create it

    log.info("Extracting features ...")
    cols = ["vocID", "audio_file", "start", "end"]
    samples = datamodule.data.filelist[cols].values
    stack = extract_feats(samples, target_sr, model, batch_size, device)

    log.info("Saving features ...")
    # sge_task_id = os.environ.get("SGE_TASK_ID")
    save_path = os.path.join(save_dir, f"all_{cfg.model}.npz")
    np.savez(save_path, **stack)

    log.info("Exiting ...")

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    # extract features
    _ = extract(cfg)


if __name__ == "__main__":
    main()
