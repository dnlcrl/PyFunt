from module import Module
import numpy as np


class Linear(Module):

    def __init__(self, input_size, output_size, bias=False):
        super(Linear, self).__init__()
        self.weight = np.ndarray((input_size, output_size))
        self.grad_weight = np.ndarray((input_size, output_size))
        if bias:
            self.bias = np.ndarray(output_size)
            self.grad_bias = np.ndarray(output_size)
        else:
            self.bias = None
            self.grad_bias = None
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
        if self.bias is not None:
            # init bias
            pass

    def update_output(self, x):
        out = x.reshape(x.shape[0], -1)
        if self.weight is not None:
            out = out.dot(self.weight)
        if self.bias is not None:
            out += self.bias
        self.output = out
        return self.output

    def update_grad_input(self, x, grad_output):
        dx = grad_output.dot(self.weight.T).reshape(x.shape)
        self.grad_weight = x.reshape(x.shape[0], -1).T.dot(grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        self.grad_input = dx
        return dx

    def acc_grad_parameters(self, grad_output, scale=None):
        pass
