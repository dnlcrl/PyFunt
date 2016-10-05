from module import Module
import numpy as np


class SoftMax(Module):
    """docstring for LogSoftMax"""
    def __init__(self):
        super(SoftMax, self).__init__()

    def update_output(self, x):
        max_input = x.max(1, keepdims=True)
        z = np.exp(x - max_input)
        log_sum = np.sum(z, axis=1, keepdims=True)
        # log_sum = max_input + np.log(log_sum)
        self.output = z * 1/log_sum
        return self.output

    def update_grad_input(self, x, grad_output):
        _sum = np.sum(grad_output*self.ouput, axis=1, keepdims=True)
        self.grad_input = self.output * (self.grad_output - _sum)

        # max_input = x.max(1, keepdims=True)
        # log_sum = np.sum(np.exp(x - max_input), axis=1, keepdims=True)
        # log_sum = max_input + np.log(log_sum)
        # self.output = x - log_sum

        # self.grad_input = grad_output - np.exp(self.output)*_sum
        return self.grad_input

    def reset(self):
        pass
