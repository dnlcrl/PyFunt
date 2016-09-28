from spatial_convolution import SpatialConvolution
from spatial_batch_normalitazion import SpatialBatchNormalization
from spatial_average_pooling import SpatialAveragePooling
from sequential import Sequential
from padding import Padding
from relu import ReLU
from c_add_table import CAddTable
from concat_table import ConcatTable


def make_residual_layer(n_channels, n_out_channels=None, stride=None):
    n_out_channels = n_out_channels or n_channels
    stride = stride or 1

    module = Sequential()
    add = module.add
    add(SpatialConvolution(
        n_channels, n_out_channels, 3, 3, stride, stride, 1, 1))
    add(SpatialBatchNormalization(n_out_channels))
    add(SpatialConvolution(n_out_channels, n_out_channels, 3, 3, 1, 1, 1, 1))

    skip = Sequential()
    if stride > 1:
        skip.add(SpatialAveragePooling(1, 1, stride, stride))
    if n_out_channels > n_channels:
        skip = skip.add(Padding(1, (n_out_channels - n_channels), 3))

    # elif n_out_channels < n_channels:
    #     skip =  skip.add(Narrow)

    add(SpatialBatchNormalization(n_out_channels))
    cat = ConcatTable()
    #net.add(cat)
    #cat.add()
    layer = CAddTable()#net, skip
    add(ReLU(True))

    return module
