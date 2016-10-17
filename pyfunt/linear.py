from module import Module
import numpy as np


class Linear(Module):

    def __init__(self, input_size, output_size, bias=True):
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
            std = stdv * np.sqrt(3)
        else:
            std = 1./np.sqrt(self.weight.shape[0])  # fan-in (th default)
            # std = 1./np.sqrt(self.weight.shape[1])  # fan-out
        self.weight = np.random.uniform(-std, std, self.weight.shape)
        # self.weight = np.random.normal(std, size=self.weight.shape)
        if self.bias is not None:
            self.bias = np.random.uniform(-std, std, self.bias.shape)
            # self.bias = np.zeros(self.bias.shape)

    def update_output(self, x):
        out = x.reshape(x.shape[0], -1)
        out = out.dot(self.weight)
        if self.bias is not None:
            out += self.bias
        self.output = out
        return self.output

    def update_grad_input(self, x, grad_output):
        dx = grad_output.dot(self.weight.T).reshape(x.shape)
        self.grad_weight[:] = x.reshape(x.shape[0], -1).T.dot(grad_output)[:]
        if self.bias is not None:
            self.grad_bias[:] = np.sum(grad_output, axis=0)[:]
        self.grad_input = dx
        return dx

    def acc_grad_parameters(self, x, grad_output, scale=None):
        pass
