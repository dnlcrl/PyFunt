#!/usr/bin/env python
# coding: utf-8

from module import Module
import numpy as np
try:
    from im2col_cyt import col2im_6d_cython
except ImportError:
    print('Installation broken, please reinstall PyFunt')


class SpatialFullConvolution(Module):

    '''implementation of layer described in https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf"'''
    n_dim = 2

    def __init__(self, n_input_plane, n_output_plane, kW, kH, dW=1, dH=1, padW=0, padH=0, adjW=0, adjH=0):
        super(SpatialFullConvolution, self).__init__()

        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.kW = kW
        self.kH = kH

        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH or self.padW
        self.adjW = adjW
        self.adjH = adjH

        if self.adjW > self.dW - 1 or self.adjH > self.dH - 1:
            raise(
                'adjW and adjH must be smaller than self.dW - 1 and self.dH - 1 respectively')

        self.weight = np.ndarray((n_input_plane, n_output_plane, kH, kW))
        self.bias = np.ndarray(n_output_plane)
        self.grad_weight = np.ndarray((n_input_plane, n_output_plane, kH, kW))
        self.grad_bias = np.ndarray(n_output_plane)

        self.reset()

    def no_bias(self):
        self.bias = None
        self.grad_bias = None

    def reset(self, stdv=None):
        if not stdv:
            stdv = stdv * np.sqrt(self.kW*self.kH*self.n_input_plane)
        else:
            stdv = stdv * np.sqrt(3)
        self.weight = np.random.normal(
            0, stdv, (self.n_output_plane, self.n_input_plane, self.kH, self.kW))
        self.bias = np.zeros(self.n_output_plane)

    def check_input_dim(self, x):
        pass

    def make_contigous(self, input, grad_output):
        pass

    def calcula_adj(self, target_size, ker, pad, stride):
        return (target_size + 2 * pad - ker) % stride

    def update_output(self, x):
        w, b = self.weight, self.bias
        # input = make_contigous (input)N, C, H, W = x.shape
        self.x_shape = N, C, H, W = x.shape
        outW = (W - 1) * self.dW - 2*self.padW + self.kW + self.adjW

        F, _, HH, WW = w.shape
        stride, pad = self.dW, self.padW

        p = pad
        x = np.pad(
            x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        self.tiles_w = (W + 2 * pad - WW) % stride
        self.tiles_h = (H + 2 * pad - HH) % stride

        w = self.weight

        stride, pad = self.dW, self.padW

        N, C, H, W = self.x_shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = x.shape

        x_reshaped = x.transpose(1, 0, 2, 3).reshape(F, -1)

        out_cols = w.reshape(F, -1).dot(x_reshaped) + b.reshape(-1, 1)
        out_cols.shape = (C, HH, WW, N, out_h, out_w)
        self.output = col2im_6d_cython(
            out_cols, N, C, H, W, HH, WW, pad, stride)
        if self.adjW:
            self.output = np.pad(
                self.output, ((0, 0), (0, 0), (0, self.adjW), (0, 0)))
        if self.adjH:
            self.output = np.pad(
                self.output, ((0, 0), (0, 0), (0, 0), (0, self.adjH)))
        assert(outW == self.output.shape[2])
        return self.output

    def update_grad_input(self, input, grad_output, scale=1):
        w = self.bias
        F, _, HH, WW = w.shape
        stride = self.stride

        if not self.adjW == 0:
            grad_output = grad_output[:, :, :-self.adjW, :]
        if not self.adjH == 0:
            grad_output = grad_output[:, :, :, :-self.adjH]

        N, C, H, W = grad_output.shape

        # H += 2 * pad
        # W += 2 * pad
        out_h = (H - HH) / stride + 1
        out_w = (W - WW) / stride + 1

        # Perform an im2col operation by picking clever strides
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = grad_output.itemsize * np.array(strides)
        dout_stride = np.lib.stride_tricks.as_strided(grad_output, shape=shape, strides=strides)
        dout_cols = np.ascontiguousarray(dout_stride)
        dout_cols.shape = (C * HH * WW, N * out_h * out_w)

        # Now all our convolutions are a big matrix multiply
        res = w.reshape(F, -1).T.dot(dout_cols)

        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)

        self.grad_input = np.ascontiguousarray(out)
        return self.grad_input

    def type(self, type, cache):
        pass

    def __str__(self):
        pass

    def clear_state(self):
        pass
