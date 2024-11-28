# Copyright (c) 2024, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>

import os

import numpy as np
import torch
import torch.nn as nn


class PANN(nn.Module):
    """Compute PANN features of a signal."""

    def __init__(self, name="pann", feats_dir=None, nan_rows=None):
        super(PANN, self).__init__()

        self.name = name
        self.feats_dir = feats_dir
        self.nan_rows = nan_rows
        self.feature_dict = dict(np.load(self.feats_dir))
        self.feature_dict = {int(k): v for k, v in self.feature_dict.items()}

        # Replace unwanted row values with the previous ones for standardization.
        # Will lower the performance by a tiny margin
        if self.nan_rows is not None:
            self.nan_rows = np.load(self.nan_rows).astype(int)
            for i in self.nan_rows:
                self.feature_dict[i] = self.feature_dict[i - 1]

    def forward(self, index):
        return torch.FloatTensor(self.feature_dict[index]).unsqueeze(0)
