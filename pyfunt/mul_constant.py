from module import Module
import numpy as np


class MulConstant(Module):

    def __init__(self, constant_scalar):
        super(MulConstant, self).__init__()
        if not np.isscalar(constant_scalar):
            raise('Constant is not a scalar: ' + constant_scalar)
        self.constant_scalar = constant_scalar

    def update_output(self, x):
        self.output = x * self.constant_scalar
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = grad_output * self.constant_scalar
        return self.grad_input

    def validate_parameters(self):
        if self.inplace:
            if self.val > self.th:
                raise('in-place processing requires value not exceed threshold')

    def reset(self):
        pass
