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


class Catch22(nn.Module):
    """Compute catch-22 features of a signal."""

    def __init__(self, name="catch22", feats_dir=None):
        super(Catch22, self).__init__()

        self.name = name
        self.feats_dir = feats_dir
        self.feature_dict = dict(np.load(self.feats_dir))
        self.feature_dict = {int(k): v for k, v in self.feature_dict.items()}

    def forward(self, index):
        return torch.FloatTensor(self.feature_dict[index]).unsqueeze(0)
