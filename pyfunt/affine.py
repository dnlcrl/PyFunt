#!/usr/bin/env python
# coding: utf-8

from module import Module
import numpy as np


class Affine(Module):

    def __init__(self, input_size, output_size, bias=False):
        super(Affine, self).__init__()
        self.weight = np.ndarray(output_size, input_size)
        self.grad_weight = np.ndarray(output_size, input_size)
        if bias:
            self.bias = np.ndarray(output_size)
            self.grad_bias = np.ndarray(output_size)
        self.reset()

    def no_bias(self):
        self.bias = None
        self.grad_bias = None
        return self

    def reset(self, stdv=None):
        if not stdv:
            stdv = 1./np.sqrt(self.weight.shape[2])
        self.weight = np.uniform(-stdv, stdv, self.weight.shape)
        self.bias = np.uniform(-stdv, stdv, self.weight.shape[0])

    def update_output(self, x):
        w = self.weight
        b = self.bias or np.zeros(self.weight.shape[0])
        self.out = x.reshape(x.shape[0], -1).dot(w) + b
        self.x = x
        return self.output

    def update_grad_input(self, input, grad_output):
        x, w = self.x, self.weight
        self.grad_input = grad_output.dot(w.T).reshape(x.shape)
        return self.grad_input

    def acc_grad_parameters(self, x, grad_output, scale=1):
        x = self.x
        self.grad_weight = x.reshape(x.shape[0], -1).T.dot(grad_output)
        if self.bias:
            self.grad_bias = np.sum(grad_output, axis=0)

    def clear_state(self):
        pass

    def __str__(self):
        pass
