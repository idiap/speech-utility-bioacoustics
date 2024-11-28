# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

# Jman/GridTK:
labels_all="calltype caller"
labels_all=($labels_all)

feats="imv_byol_16000 imv_c22_16000 imv_wavlm_16000 imv_pann8000 imv_pann16000 imv_pann32000"
feats=($feats)

for ft in "${feats[@]}"; do
    for l in "${labels_all[@]}"; do
        jman -vv submit -q sgpu -s TMP TEMP TMPDIR SGE_TEMPROOT MPLCONFIGDIR WANDB_CACHE_DIR TRANSFORMERS_CACHE PYTHONUNBUFFERED=1 -n pdist_${ft}_${l} -- `which python` src/pdists.py experiment=pdists/${ft}.yaml +data.data.selected_labels=${l}
        sleep 1
    done
done