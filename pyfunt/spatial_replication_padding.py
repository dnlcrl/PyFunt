from module import Module
import numpy as np


class SpatialReplicationPadding(Module):

    def __init__(self, pad_l, pad_r=None, pad_t=None, pad_b=None):
        super(SpatialReplicationPadding, self).__init__()
        self.pad_l = pad_l
        self.pad_r = pad_r or self.pad_l
        self.pad_t = pad_t or self.pad_l
        self.pad_b = pad_b or self.pad_l

    def update_output(self, x):
        if x.dims == 3:
            self.output = np.pad(
                x, ((0, 0), (self.pad_t, self.pad_b), (self.pad_l, self.pad_r)), 'edge')
        elif x.dims == 4:
            self.output = np.pad(
                x, ((0, 0), (0, 0), (self.pad_t, self.pad_b), (self.pad_l, self.pad_r)), 'edge')

        else:
            raise('input must be 3 or 4-dimensional')
        return self.output

    def update_grad_input(self, x, grad_output):
        if x.dims == grad_output.dims == 3:
            if not (x.shape[0] == grad_output.shape[0] and
                    x.shape[1] + self.pad_t + self.pad_b == grad_output.shape[1] and
                    x.shape[2] + self.pad_l + self.pad_r == grad_output.shape[2]):
                raise('input and gradOutput must be compatible in size')
            self.grad_input = grad_output[:, self.pad_t:self.pad_b, self.pad_l:self.pad_r]
        elif x.dims == grad_output.dims == 4:
            if not (x.shape[0] == grad_output.shape[0] and
                    x.shape[1] == grad_output.shape[1] and
                    x.shape[2] + self.pad_t + self.pad_b == grad_output.shape[2] and
                    x.shape[3] + self.pad_l + self.pad_r == grad_output.shape[3]):
                raise('input and gradOutput must be compatible in size')
            self.grad_input = grad_output[:, :, self.pad_t:self.pad_b, self.pad_l:self.pad_r]
        else:
            raise(
                'input and gradOutput must be 3 or 4-dimensional and have equal number of dimensions')
        return self.grad_input

    def __str__(self):
        return type(self) + '(l=%d, r=%d, t=%d, b=%d)' % (self.pad_l, self.pad_r, self.pad_t, self.pad_b)
