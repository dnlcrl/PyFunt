from module import Module
import numpy as np


class Sigmoid(Module):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def update_output(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = grad_output * (1.0 - grad_output)
        return self.grad_input
