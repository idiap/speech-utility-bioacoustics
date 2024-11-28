# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

# Jman/GridTK:
bs_all="32 64 128 256 512"
bs_all=($bs_all)

lr_all="0.001 0.0001"
lr_all=($lr_all)

for bs in "${bs_all[@]}"; do
    for lr in "${lr_all[@]}"; do
        # imv
        jman -vv submit -q sgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_imv_calltype_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/imv.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        jman -vv submit -q sgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_imv_caller_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/imv.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        # Alex
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_alex_calltype_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/alex.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_alex_caller_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/alex.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_alex_gender_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/alex.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        # Kaja
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_kaja_calltype_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/kaja.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_kaja_caller_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/kaja.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
        jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n c22_kaja_gender_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_c22_16000/kaja.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr
        sleep 1
    done
done