from container import Container
import numpy as np


class Sequential(Container):

    """docstring for Sequential"""

    def __init__(self):
        super(Sequential, self).__init__()

    def len(self):
        return len(self.modules)

    def add(self,  module):
        if len(self.modules) == 0:
            self.grad_input = module.grad_input
        self.modules.append(module)
        self.output = module.output
        return self

    def insert(self, module, index=None):
        index = index or len(self.modules) + 1
        if index > len(self.modules) + 1 or index < 1:
            raise Exception('index should be contiguous to existing modules')
        self.modules.insert(module, index)
        self.output = self.modules[len(self.modules)].output
        self.grad_input = self.modules[0].grad_input  # 1??

    def remove(self, index):
        if index > len(self.modules) or index < 1:
            raise Exception('index out of range')
        self.modules.remove(index)
        if len(self.modules) > 0:
            self.output = self.modules[-1].output
            self.grad_input = self.modules[0].grad_input
        else:
            self.output = np.ndarray()
            self.grad_input = np.ndarray()

    def update_output(self, x):
        current_output = x
        for i in xrange(len(self.modules)):
            current_output = self.rethrow_errors(self.modules[i], i, 'update_output', current_output)
        self.output = current_output
        return self.output

    def update_grad_input(self, x, grad_output):
        current_grad_output = grad_output
        current_module = self.modules[-1]
        for i in range(len(self.modules)-2, -1, -1):
            previous_module = self.modules[i]
            current_grad_output = self.rethrow_errors(current_module, i, 'update_grad_input', previous_module.output, current_grad_output)
            current_module = previous_module
        current_grad_output = self.rethrow_errors(current_module, 0, 'update_grad_input', x, current_grad_output)
        self.grad_input = current_grad_output
        return current_grad_output

    def acc_grad_parameters(self, x, grad_output, scale=1):
        current_grad_output = grad_output
        current_module = self.modules[-1]
        for i in range(len(self.modules)-2, -1, -1):
            previous_module = self.modules[i]
            self.rethrow_errors(current_module, i, 'acc_grad_parameters', previous_module.output, current_grad_output, scale)
            current_grad_output = current_module.grad_input
            current_module = previous_module
        self.rethrow_errors(current_module, 0, 'acc_grad_parameters', x, current_grad_output, scale)

    def backward(self, x, grad_output, scale=1):
        current_grad_output = grad_output
        current_module = self.modules[-1]
        for i in range(len(self.modules)-2, -1, -1):
            previous_module = self.modules[i]
            current_grad_output = self.rethrow_errors(current_module, i, 'backward', previous_module.output, current_grad_output, scale)
            current_module.grad_input[:] = current_grad_output[:]
            current_module = previous_module

        current_grad_output = self.rethrow_errors(current_module, 0, 'backward', x, current_grad_output, scale)
        self.grad_input = current_grad_output
        return current_grad_output

    def acc_update_grad_parameters(self, x, grad_output, lr):
        current_grad_output = grad_output
        current_module = self.modules[-1]
        for i in range(len(self.modules)-2, -1, -1):
            previous_module = self.modules[i]
            self.rethrow_errors(current_module, i, 'acc_update_grad_parameters', previous_module.output, current_grad_output, lr)
            current_grad_output = current_module.grad_input
            current_module = previous_module
        self.rethrow_errors(current_module, 1, 'acc_update_grad_parameters', x, current_grad_output, lr)

    def __str__(self):
        return 'temporary string for Sequential class'
