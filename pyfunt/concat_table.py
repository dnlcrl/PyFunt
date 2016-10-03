from container import Container
import numpy as np


class ConcatTable(Container):

    """docstring for ConcatTable"""

    def __init__(self):
        super(ConcatTable, self).__init__()
        self.modules = []

    def update_output(self, x):
        self.output = []
        for i in xrange(len(self.modules)):
            current_output = self.rethrow_errors(
                self.modules[i], i, 'update_output', x)
            self.output.append(current_output)
            # if i == 0:
            #     self.output = current_output
            # else:
            #     np.concatenate((self.output, current_output), axis=0)
            # self.output.append(self.rethrow_errors(
            # self.modules[i], i, 'update_output', x))
        return self.output

    def _backward(self, method, x, grad_output, scale):
        for i, module in enumerate(self.modules):
            current_grad_input = self.rethrow_errors(
                self.modules[i], i, method, x, grad_output[i], scale)
            if i == 0:
                self.grad_input = current_grad_input
            else:
                self.grad_input += current_grad_input
        return self.grad_input

    def update_grad_input(self, x, grad_output):
        return self._backward('update_grad_input', x, grad_output)

    def backward(self, x, grad_output, scale=1):
        return self._backward('backward', x, grad_output, scale)

    def acc_grad_parameters(self, x, grad_output, scale=1):
        for i, module in enumerate(self.modules):
            self.rethrow_errors(
                self.modules[i], i, 'acc_grad_parameters', x, grad_output[i], scale)
