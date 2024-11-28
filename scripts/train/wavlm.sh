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

l_all="0 1 2 3 4" #5 6 7 8 9 10 11 12
l_all=($l_all)

for bs in "${bs_all[@]}"; do
    for lr in "${lr_all[@]}"; do
        for l in "${l_all[@]}"; do
            # imv
            jman -vv submit -q sgpu -l -e="'hostname=!vgni*'" -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_imv_calltype_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/imv.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            sleep 1
            jman -vv submit -q sgpu -l -e="'hostname=!vgni*'" -s TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_imv_caller_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/imv.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            sleep 1
            # # Alex
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_alex_calltype_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/alex.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_alex_caller_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/alex.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_alex_gender_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/alex.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
            # # Kaja
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_kaja_calltype_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/kaja.yaml trainer=gpu +data.data.selected_labels=calltype data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_kaja_caller_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/kaja.yaml trainer=gpu +data.data.selected_labels=caller data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
            # jman -vv submit -q vsgpu TMP TEMP TMPDIR SGE_TEMPROOT WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n wavlm_kaja_gender_bs${bs}_lr${lr}_layer${l} -- `which python` src/train.py experiment=train_wavlm_16000/kaja.yaml trainer=gpu +data.data.selected_labels=gender data.batch_size=$bs model.optimizer.lr=$lr +layer_id=$l
            # sleep 1
        done
    done
done