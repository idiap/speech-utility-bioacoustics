# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

# Dask version (crashes):
#python src/train.py -m experiment=train_pann_32000/alex.yaml,train_pann_32000/imv.yaml,train_pann_32000/kaja.yaml,train_pann_16000/alex.yaml,train_pann_16000/imv.yaml,train_pann_16000/kaja.yaml,train_pann_8000/alex.yaml,train_pann_8000/imv.yaml,train_pann_8000/kaja.yaml model.optimizer.lr=1e-3,1e-4 data.batch_size=32,64,128,256,512 +data.data.selected_labels=calltype,caller +dask=vsgpu

# Jman/GridTK:
bs_all="32 64 128 256 512"
bs_all=($bs_all)

lr_all="0.001 0.0001"
lr_all=($lr_all)

sr_all="32000 16000 8000"
sr_all=($sr_all)

for bs in "${bs_all[@]}"; do
    for lr in "${lr_all[@]}"; do
        for sr in "${sr_all[@]}"; do
            # imv
            jman -vv submit -q sgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_imv_calltype_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/imv.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            jman -vv submit -q sgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_imv_caller_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/imv.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            # Alex
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_alex_calltype_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/alex.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_alex_caller_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/alex.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_alex_gender_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/alex.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            # Kaja
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_kaja_calltype_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/kaja.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_kaja_caller_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/kaja.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
            jman -vv submit -q vsgpu -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pann_kaja_gender_sr${sr}_bs${bs}_lr${lr} -- `which python` src/train.py experiment=train_pann_${sr}/kaja.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr
            sleep 1
        done
    done
done