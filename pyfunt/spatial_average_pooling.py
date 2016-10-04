from module import Module
import numpy as np

try:
    from im2col_cyt import im2col_cython, col2im_cython
except ImportError:
    print('Installation broken, please reinstall PyFunt')


class SpatialAveragePooling(Module):

    """docstring for SpatialAveragePooling"""

    def __init__(self, kW, kH, dW=1, dH=1, padW=0, padH=0):
        super(SpatialAveragePooling, self).__init__()
        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH
        self.ceil_mode = False
        self.count_include_pad = True
        self.divide = True

    def reset(self):
        #TODO
        pass

    def ceil(self):
        self.ceil_mode = True

    def floor(self):
        self.ceil_mode = False

    def set_count_include_pad(self):
        self.count_include_pad = True

    def set_count_exclude_pad(self):
        self.count_include_pad = False

    def update_output(self, x):
        N, C, H, W = x.shape
        pool_height, pool_width = self.kW, self.kH
        stride = self.dW

        assert (
            H - pool_height) % stride == 0 or H == pool_height, 'Invalid height'
        assert (
            W - pool_width) % stride == 0 or W == pool_width, 'Invalid width'

        out_height = int(np.floor((H - pool_height) / stride + 1))
        out_width = int(np.floor((W - pool_width) / stride + 1))

        x_split = x.reshape(N * C, 1, H, W)
        x_cols = im2col_cython(
            x_split, pool_height, pool_width, padding=0, stride=stride)
        x_cols_avg = np.mean(x_cols, axis=0)
        out = x_cols_avg.reshape(
            out_height, out_width, N, C).transpose(2, 3, 0, 1)

        self.x_shape = x.shape
        self.x_cols = x_cols
        self.output = out
        return self.output

    def update_grad_input(self, x, grad_output, scale=1):
        x_cols = self.x_cols
        dout = grad_output
        N, C, H, W = self.x_shape
        pool_height, pool_width = self.kW, self.kH
        stride = self.dW
        pool_dim = pool_height * pool_width

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[:, np.arange(dx_cols.shape[1])] = 1. / pool_dim * dout_reshaped
        dx = col2im_cython(dx_cols, N * C, 1, H, W, pool_height, pool_width,
                           padding=0, stride=stride)

        self.grad_input = dx.reshape(self.x_shape)

        return self.grad_input

    def __str__(self):
        pass
