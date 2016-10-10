#!/usr/bin/env python
# coding: utf-8

from module import Module
import numpy as np


class BatchNormalization(Module):

    def __init__(self, n_output, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNormalization, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.train = True
        self.running_mean = np.zeros(n_output)
        self.running_var = np.zeros(n_output)
        n_dim = 2
        if affine:
            self.weight = np.ndarray(n_output)
            self.bias = np.ndarray(n_output)
            self.grad_weight = np.ndarray(n_output)
            self.grad_bias = np.ndarray(n_output)
        else:
            self.weight = None
            self.bias = None
            self.grad_weight = None
            self.grad_bias = None
        self.reset()

    def reset(self):
        if self.weight is not None:
            self.weight[:] = np.random.uniform(size=len(self.weight))[:]
        if self.bias is not None:
            self.bias[:] = np.zeros(len(self.bias))[:]
        self.running_mean = np.zeros(len(self.running_mean))
        self.running_var = np.ones(len(self.running_var))

    def check_input_dim(self, x):
        i_dim = len(x.shape)
        if i_dim != self.n_dim or (i_dim != self.n_dim - 1 and self.train is not False):
            raise Exception('TODO ERROR :(')
        # feast_dim = (i_dim == self.n_dim -1) and 1 or 2
        #      local featDim = (iDim == self.nDim - 1) and 1 or 2
        # assert(input:size(featDim) == self.running_mean:nElement(), string.format(
        #    'got %d-feature tensor, expected %d',
   #    input:size(featDim), self.running_mean:nElement()))

    def make_contigous(self, x, grad_output):
        #TODO
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

            running_var = momentum * unbiased_var + \
                (1. - momentum) * running_var

            self.xmu = xmu
            self.invstd = invstd

        else:
            mean = running_mean
            invstd = 1. / np.sqrt(running_var + eps)

        out = ((x - mean) * invstd)
        if self.weight is not None:
            out *= self.weight
        if self.bias is not None:
            out += self.bias
        #out = ((x - mean) * invstd) * self.weight + self.bias
        # Store the updated running means back into bn_param
        self.running_mean = np.array(running_mean, copy=True)
        self.running_var = np.array(running_var, copy=True)
        self.output = out

        return self.output

    def update_grad_input(self, x, grad_output, scale=1):

        xmu, invstd = self.xmu, self.invstd

        N, D = grad_output.shape

        _sum = np.sum(grad_output, axis=0)
        dotp = np.sum((xmu * grad_output), axis=0)

        k = 1. / N * dotp * invstd * invstd
        dx = xmu * k

        dmean = 1. / N * _sum
        dx = (grad_output - dmean - dx) * invstd * self.weight

        self.grad_weight[:] = dotp * invstd

        self.grad_bias[:] = _sum
        self.grad_input = dx

        return self.grad_input

    # def backward(self, x, grad_output, scale=1):
    #     return self.update_grad_input(x, grad_output, scale)

    def acc_grad_input(self, x, grad_output, scale):
        return self.backward(x, grad_output, scale, None, self.grad_weight, self.grad_bias)

    def clear_state(self):
        pass
