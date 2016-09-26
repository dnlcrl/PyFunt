from module import Module
import numpy as np


class Linear(Module):

    def __init__(self, input_size, output_size, bias=False):
        super(Linear, self).__init__()
        self.weight = np.ndarray(output_size, input_size)
        self.grad_weight = np.ndarray(output_size, input_size)
        if bias:
            self.bias = np.ndarray(output_size)
            self.grad_bias = np.ndarray(output_size)
        self.reset()

    def no_bias(self):
        self.bias = None
        self.grad_bias = None

    def reset(self, stdv=None):
        if stdv:
            stdv = stdv * np.sqrt(3)
        else:
            stdv = 1./np.sqrt(self.weight.shape[1])
        # init weight
        if self.bias:
            # init bias

    def update_output(self, x):
        pass

    def updaet_grad_input(self, grad_output):
        pass

    def acc_grad_parameters(self, grad_output, scale=None):
        pass
