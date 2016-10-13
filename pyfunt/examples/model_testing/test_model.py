import numpy as np
from pyfunt import (SpatialConvolution, SpatialBatchNormalization,
                    SpatialAveragePooling, Sequential, ReLU, Linear,
                    Reshape, LogSoftMax)
from pyfunt.utils import eval_numerical_gradient_array

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

x = np.random.randn(3, 4, 8, 8)
# x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 10)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

s = Sequential()
s.add(SpatialConvolution(4, 2, 1, 1, 1, 1))
s.add(SpatialAveragePooling(2, 2, 2, 2, 0, 0))
s.add(SpatialBatchNormalization(2))
s.add(ReLU())
s.add(Reshape(2*4*4))
s.add(Linear(2*4*4, 10))
s.add(LogSoftMax())

dx_num = eval_numerical_gradient_array(lambda x: s.update_output(x), x, dout)

out = s.update_output(x)
dx = s.update_grad_input(x, dout)
# Your error should be around 1e-8
print('Testing net backward function:')
print('dx error: ', rel_error(dx, dx_num))
# import pdb; pdb.set_trace()
