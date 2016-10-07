from module import Module
import numpy as np


class Tanh(Module):

    def __init__(self, th=1e-6, v=0, ip=False):
        super(Tanh, self).__init__()
        self.th = th
        self.val = v
        self.inplace = ip

    def update_output(self, x):
        self.output = np.tanh(x)
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = grad_output * (1 - np.power(self.output, 2))
        return self.grad_input
