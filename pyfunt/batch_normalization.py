#!/usr/bin/env python
# coding: utf-8

from module import Module
import numpy as np


class BatchNormalization(Module):

    def __init__(self, n_output, eps=1e-5, momentum=0.1, affine=False):
        super(BatchNormalization, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.train = True
        self.running_mean = np.zeros(n_output)
        self.running_var = np.zeros(n_output)
        if affine:
            self.weight = np.ndarray(n_output)
            self.bias = np.ndarray(n_output)
            self.grad_weight = np.ndarray(n_output)
            self.grad_bias = np.ndarray(n_output)
            self.reset()

    def reset(self):
        if self.weight:
            self.weight = np.uniform(len(self.weight))
        if self.bias:
            self.bias = np.zeros(len(self.weight))
        self.running_mean = np.zeros(len(self.weight))
        self.running_var = np.ones(len(self.weight))

    def check_input_dim(self):
        pass

    def make_contigous(self):
        pass

    def update_output(self, x):

        eps = self.eps
        momentum = self.momentum
        N, D = x.shape
        running_mean = self.running_mean
        running_var = self.running_var

        if self.train:
            mean = 1. / N * np.sum(x, axis=0)

            xmu = x - mean

            carre = xmu*xmu

            var = 1. / N * np.sum(carre, axis=0)

            sqrtvar = np.sqrt(var + eps)

            invstd = 1. / sqrtvar

            running_mean = momentum * mean + (1. - momentum) * running_mean

            unbiased_var = np.sum(carre, axis=0)/(N - 1.)

            running_var = momentum * unbiased_var + (1. - momentum) * running_var

            self.xmu = xmu
            self.invstd = invstd

        else:
            mean = running_mean
            invstd = 1. / np.sqrt(running_var + eps)

        out = ((x - mean) * invstd) * self.weight + self.bias
        # Store the updated running means back into bn_param
        self.running_mean = np.array(running_mean, copy=True)
        self.running_var = np.array(running_var, copy=True)
        self.output = out

        return self.output

    def backward(self, x, grad_output, scale, grad_input, grad_weight=None, grad_bias=None):

        xmu, invstd = self.xmu, self.invstd

        N, D = grad_output.shape

        _sum = np.sum(grad_output, axis=0)
        dotp = np.sum((xmu * grad_output), axis=0)

        k = 1. / N * dotp * invstd * invstd
        dx = xmu * k

        dmean = 1. / N * _sum
        dx = (grad_output - dmean - dx) * invstd * self.weight

        self.grad_weight = dotp * invstd

        self.grad_bias = _sum

        return dx, self.grad_weight, self.grad_bias

    def update_grad_input(self, x, grad_output):
        return self.backward(x, grad_output, 1, self.grad_input)

    def acc_grad_input(self, x, grad_output, scale):
        return self.backward(x, grad_output, scale, None, self.grad_weight, self.grad_bias)

    def clear_state(self):
        pass
