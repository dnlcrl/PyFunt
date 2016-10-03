from spatial_convolution import SpatialConvolution
from spatial_batch_normalitazion import SpatialBatchNormalization
from spatial_average_pooling import SpatialAveragePooling
from sequential import Sequential
from padding import Padding
from relu import ReLU
from c_add_table import CAddTable
from concat_table import ConcatTable
from identity import Identity


def residual_layer(n_channels, n_out_channels=None, stride=None):
    n_out_channels = n_out_channels or n_channels
    stride = stride or 1

    convs = Sequential()
    add = convs.add
    add(SpatialConvolution(
        n_channels, n_out_channels, 3, 3, stride, stride, 1, 1))
    add(SpatialBatchNormalization(n_out_channels))
    add(SpatialConvolution(n_out_channels, n_out_channels, 3, 3, 1, 1, 1, 1))
    add(SpatialBatchNormalization(n_out_channels))

    if stride > 1:
        shortcut = Sequential()
        shortcut.add(SpatialAveragePooling(2, 2, stride, stride))
        shortcut.add(Padding(1, (n_out_channels - n_channels)/2, 3))
    else:
        shortcut = Identity()

    res = Sequential()
    res.add(ConcatTable().add(convs).add(shortcut)).add(CAddTable())
    # https://github.com/szagoruyko/wide-residual-networks/blob/master/models/resnet-pre-act.lua

    res.add(ReLU(True))

    return res
