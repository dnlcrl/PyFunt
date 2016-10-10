#!/usr/bin/env python
# coding: utf-8
from module import Module
import numpy as np
try:
    from im2col_cyt import im2col_cython, col2im_cython
except ImportError:
    print('Installation broken, please reinstall PyFunt')

from numpy.lib.stride_tricks import as_strided


def tile_array(a, b1, b2):
    r, c = a.shape
    rs, cs = a.strides
    x = as_strided(a, (r, b1, c, b2), (rs, 0, cs, 0))
    return x.reshape(r*b1, c*b2)


class SpatialUpSamplingNearest(Module):

    def __init__(self, scale):
        super(SpatialUpSamplingNearest, self).__init__()
        self.scale_factor = scale
        if self.scale_factor < 1:
            raise Exception('scale_factor must be greater than 1')
        if np.floor(self.scale_factor) != self.scale_factor:
            raise Exception('scale_factor must be integer')

    def update_output(self, x):
        out_size = x.shape
        out_size[x.ndim - 1] *= self.scale_factor
        out_size[x.ndim - 2] *= self.scale_factor
        N, C, H, W = out_size

        stride = self.scale_factor
        pool_height = pool_width = stride

        x_reshaped = x.transpose(2, 3, 0, 1).flatten()
        out_cols = np.zeros(out_size)
        out_cols[:, np.arange(out_cols.shape[1])] = x_reshaped
        out = col2im_cython(out_cols, N * C, 1, H, W, pool_height, pool_width,
                            padding=0, stride=stride)
        out = out.reshape(out_size)
        return self.grad_input

        return self.output

    def update_grad_input(self, x, grad_output, scale=1):

        N, C, H, W = grad_output.shape
        pool_height = pool_width = self.scale_factor
        stride = self.scale_factor

        out_height = (H - pool_height) / stride + 1
        out_width = (W - pool_width) / stride + 1

        grad_output_split = grad_output.reshape(N * C, 1, H, W)
        grad_output_cols = im2col_cython(
            grad_output_split, pool_height, pool_width, padding=0, stride=stride)
        grad_intput_cols = grad_output_cols[0, np.arange(grad_output_cols.shape[1])]
        grad_input = grad_intput_cols.reshape(
            out_height, out_width, N, C).transpose(2, 3, 0, 1)

        self.output = grad_input



