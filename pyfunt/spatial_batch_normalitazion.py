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

    def update_grad_input(self, x, grad_output, scale=1):
        N, C, H, W = grad_output.shape
        dout_flat = grad_output.transpose(0, 2, 3, 1).reshape(-1, C)
        dout_flat = np.ascontiguousarray(dout_flat, dtype=dout_flat.dtype)
        super(SpatialBatchNormalization, self).update_grad_input(x, dout_flat, scale)
        self.grad_input = self.grad_input.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return self.grad_input

    def backward(self, x, grad_output, scale=1):
        return self.update_grad_input(x, grad_output, scale)
