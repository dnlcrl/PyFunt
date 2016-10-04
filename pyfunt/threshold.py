from module import Module
import numpy as np


class Threshold(Module):

    def __init__(self, th=1e-6, v=0, ip=False):
        super(Threshold, self).__init__()
        self.th = th
        self.val = v
        self.inplace = ip

    def update_output(self, x):
        #out = x
        #indices = np.where(x > self.th)
        #self.indices = indices
        self.output = np.maximum(self.th, x)
        return self.output

    def update_grad_input(self, x, grad_output, scale=1):
        #self.validate_parameters()
        # indices = self.indices
        # if self.inplace:
        #     grad_output[indices] = 0.
        #     return grad_output
        # dx = grad_output
        # dx[indices] = 0.
        # self.grad_input = dx
        # return self.grad_input
        dx = np.array(grad_output, copy=True)
        dx[x <= 0] = 0
        self.grad_input = dx
        return self.grad_input

    def validate_parameters(self):
        if self.inplace:
            if self.val > self.th:
                raise('in-place processing requires value not exceed threshold')

    def reset(self):
        pass
