from module import Module
import numpy as np


class SpatialMaxPooling(Module):
    """docstring for SpatialMaxPooling"""
    def __init__(self, kW, kH, dW, dH, padW, padH):
        super(SpatialMaxPooling, self).__init__()
        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH
        self.ceil_mode = False
        self.count_include_pad = True
        self.divide = True

    def ceil(self):
        self.ceil_mode = True

    def floor(self):
        self.ceil_mode = False

    def set_count_include_pad(self):
        self.count_include_pad = True

    def set_count_exclude_pad(self):
        self.count_include_pad = False

    def update_output(self, x):
        return self.output

    def update_update_grad_inpu(self, x, grad_output):
        return self.grad_input

    def __str__(self):
        pass
