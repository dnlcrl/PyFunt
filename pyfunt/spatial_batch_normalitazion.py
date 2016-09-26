#!/usr/bin/env python
# coding: utf-8

from batch_normalization import BatchNormalization


class SpatialBatchNormalization(BatchNormalization):
    n_dim = 4

    def __init__(self, args):
        super(SpatialBatchNormalization, self).__init__(self, args)
