#!/usr/bin/env python
# coding: utf-8
import numpy as np
from batch_normalization import BatchNormalization


class SpatialBatchNormalization(BatchNormalization):
    n_dim = 4

    def __init__(self, args):
        super(SpatialBatchNormalization, self).__init__(args)

    def update_output(self, x):
        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        x_flat = np.ascontiguousarray(x_flat, dtype=x.dtype)
        super(SpatialBatchNormalization, self).update_output(x_flat)
        self.output = self.output.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return self.output

    def backward(self, x, grad_output, scale, grad_input, grad_weight=None, grad_bias=None):
        N, C, H, W = grad_output.shape
        dout_flat = grad_output.transpose(0, 2, 3, 1).reshape(-1, C)
        dout_flat = np.ascontiguousarray(dout_flat, dtype=dout_flat.dtype)
        super(SpatialBatchNormalization, self).backward(x, dout_flat, scale, grad_input, grad_weight, grad_bias)
        self.grad_input = self.grad_input.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return self.grad_input, self.grad_weight, self.grad_bias
