from module import Module
import numpy as np


class LogSoftMax(Module):
    """docstring for LogSoftMax"""
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def update_output(self, x):
        max_input = x.max(1, keepdims=True)
        log_sum = np.sum(np.exp(x - max_input), axis=1, keepdims=True)
        log_sum = max_input + np.log(log_sum)
        self.output = x - log_sum
        return self.output

    def update_grad_input(self, x, grad_output):
        _sum = np.sum(grad_output, axis=1, keepdims=True)
        self.grad_input = grad_output - np.exp(self.output)*_sum

        return self.grad_input

    def reset(self):
        pass
