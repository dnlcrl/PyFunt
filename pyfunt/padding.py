from module import Module
import numpy as np


class Padding(Module):

    def __init__(self, dim, pad, n_input_dim, value=None, index=None):
        super(Padding, self).__init__()
        self.value = value or 0
        self.index = index or 1
        self.dim = dim
        self.pad = pad
        self.n_input_dim = n_input_dim
        # self.output_size = longStorage

    def update_output(self, x):
        #pad
        return self.output

    def upgrade_grad_input(self, x, grad_output):
        #unpad
        return self.grad_input
