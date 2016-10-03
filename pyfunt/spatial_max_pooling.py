from module import Module
import numpy as np
try:
    from im2col_cyt import im2col_cython, col2im_cython
except ImportError:
    print 'Installation broken, please reinstall PyFunt'


class SpatialMaxPooling(Module):

    """docstring for SpatialMaxPooling"""

    def __init__(self, kW, kH, dW=1, dH=1, padW=0, padH=0):
        super(SpatialMaxPooling, self).__init__()
        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH
        self.ceil_mode = False
        self.count_include_pad = True
        self.divide = True

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

        assert (H - pool_height) % stride == 0, 'Invalid height'
        assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = (H - pool_height) / stride + 1
        out_width = (W - pool_width) / stride + 1

        x_split = x.reshape(N * C, 1, H, W)
        x_cols = im2col_cython(
            x_split, pool_height, pool_width, padding=0, stride=stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(
            out_height, out_width, N, C).transpose(2, 3, 0, 1)

        self.x_shape = x.shape
        self.x_cols = x_cols
        self.x_cols_argmax = x_cols_argmax
        self.output = out
        return self.output

    def update_grad_input(self, x, grad_output, scale=1):
        x_cols = self.x_cols
        x_cols_argmax = self.x_cols_argmax
        dout = grad_output
        N, C, H, W = x.shape
        pool_height, pool_width = self.kW, self.kH
        stride = self.dW

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = col2im_cython(dx_cols, N * C, 1, H, W, pool_height, pool_width,
                           padding=0, stride=stride)
        dx = dx.reshape(self.x_shape)
        self.grad_input = dx
        return self.grad_input

    def reset(self):
        pass

    def __str__(self):
        pass
