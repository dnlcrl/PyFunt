from module import Module
import numpy as np


class CAddTable(Module):

    def __init__(self):
        super(CAddTable, self).__init__()
        self.grad_input = None

    def update_output(self, x):
        self.output = np.sum(x, axis=0)
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = np.zeros_like(x)
        for i in xrange(len(x)):
            self.grad_input[i] = np.copy(grad_output)
        return self.grad_input

    def reset(self):
        pass
