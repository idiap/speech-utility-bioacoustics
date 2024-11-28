# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

python src/extract_pann.py -m experiment=extract_pann_32000/alex.yaml,extract_pann_32000/imv.yaml,extract_pann_32000/kaja.yaml,extract_pann_16000/alex.yaml,extract_pann_16000/imv.yaml,extract_pann_16000/kaja.yaml,extract_pann_8000/alex.yaml,extract_pann_8000/imv.yaml,extract_pann_8000/kaja.yaml +dask=vsgpu