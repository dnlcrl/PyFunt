from module import Module
import numpy as np


class Reshape(Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        if type(shape) is not tuple:
            shape = (shape,)
        self.shape = shape

    def update_output(self, x):
        self.output = x.reshape((x.shape[0],) + self.shape)
        return self.output

    def upgrade_grad_input(self, x, grad_output):
        self.grad_input = grad_output.reshape(x.shape)
        return self.grad_input
