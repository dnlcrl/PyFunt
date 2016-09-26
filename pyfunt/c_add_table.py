from module import Module
import numpy as np


class CAddTable(Module):

    def __init__(self):
        super(CAddTable, self).__init__()
        self.grad_input = {}

    def update_output(self, x):
        self.output = np.sum(x, axis=0)
        return self.output

    def upgrade_grad_input(self, x, grad_output):
        for i in xrange(len(x)):
            self.grad_input[i] = np.copy(grad_output)
        return self.grad_input
