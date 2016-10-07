from module import Module
import numpy as np


class Padding(Module):

    def __init__(self, dim, pad, n_input_dim, value=None, index=None):
        super(Padding, self).__init__()
        self.value = value or 0
        self.index = index or 1
        self.dim = [dim] if type(dim) == int else dim
        self.pad = pad if pad > 0 else -pad
        self.n_input_dim = n_input_dim

    def update_output(self, x):
        pads = []
        for axis in range(x.ndim):
            if axis in self.dim:
                pads += [(self.pad, self.pad)]
            else:
                pads += [(0, 0)]
        pads = tuple(pads)
        self.output = np.pad(x, pads, mode='constant')
        return self.output

    def update_grad_input(self, x, grad_output):
        slc = [slice(None)] * x.ndim
        self.grad_input = grad_output
        for axis in range(x.ndim):
            if axis in self.dim:
                slc[axis] = slice(self.pad, -self.pad)
        self.grad_input = grad_output[slc]
        return self.grad_input

    def reset(self):
        pass
