#!/usr/bin/env python
# coding: utf-8

from module import Module
import numpy as np
try:
    from im2col_cyt import col2im_6d_cython
except ImportError:
    print 'Installation broken, please reinstall PyFunt'


class SpatialConvolution(Module):

    n_dim = 2

    def __init__(self, n_input_plane, n_output_plane, kW, kH, dW=1, dH=1, padW=0, padH=None):
        super(SpatialConvolution, self).__init__()

        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.kW = kW
        self.kH = kH

        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH or self.padW

        self.weight = np.ndarray((n_output_plane, n_input_plane, kH, kW))
        self.bias = np.ndarray(n_output_plane)
        self.grad_weight = np.ndarray((n_output_plane, n_input_plane, kH, kW))
        self.grad_bias = np.ndarray(n_output_plane)

        self.reset()

    def no_bias(self):
        self.bias = None
        self.grad_bias = None

    def reset(self, stdv=None):
        if not stdv:
            stdv = 1/np.sqrt(self.kW*self.kH*self.n_input_plane)
        self.weight = np.random.normal(
            0, stdv, (self.n_output_plane, self.n_input_plane, self.kH, self.kW))
        self.bias = np.zeros(self.n_output_plane)

    def check_input_dim(self, x):
        pass

    def make_contigous(self, input, grad_output):
        pass

    def update_output(self, x):
        w, b = self.weight, self.bias
        # input = make_contigous (input)N, C, H, W = x.shape
        self.x_shape = N, C, H, W = x.shape

        F, _, HH, WW = w.shape
        stride, pad = self.dW, self.padW
        #assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
        #assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

        p = pad
        x_padded = np.pad(
            x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        self.tiles_w = (W + 2 * pad - WW) % stride
        self.tiles_h = (H + 2 * pad - HH) % stride
        if not self.tiles_w == 0:
            x_padded = x_padded[:, :, :-self.tiles_w, :]
        if not self.tiles_h == 0:
            x_padded = x_padded[:, :, :, :-self.tiles_h]

        N, C, H, W = x_padded.shape
        assert (W + 2 * pad - WW) % stride == 0, 'width does not work'

        # H += 2 * pad
        # W += 2 * pad
        out_h = (H - HH) / stride + 1
        out_w = (W - WW) / stride + 1

        # Perform an im2col operation by picking clever strides
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                                   shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * out_h * out_w)

        # Now all our convolutions are a big matrix multiply
        res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)

        self.output = np.ascontiguousarray(out)

        self.x_cols = x_cols
        return self.output

    def update_grad_input(self, input, grad_output, scale):
        x_shape, x_cols = self.x_shape, self.x_cols
        w = self.weight

        stride, pad = self.dW, self.padW

        N, C, H, W = x_shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = grad_output.shape

        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        dout_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(F, -1)
        self.grad_weight = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

        return dx

    def type(self, type, cache):
        pass

    def __str__(self):
        pass

    def clear_state(self):
        pass
