from pyfunt.spatial_convolution import SpatialConvolution
from pyfunt.spatial_batch_normalitazion import SpatialBatchNormalization
from pyfunt.spatial_average_pooling import SpatialAveragePooling
from pyfunt.sequential import Sequential
from pyfunt.relu import ReLU
from pyfunt.linear import Linear
from pyfunt.reshape import Reshape
from pyfunt.log_soft_max import LogSoftMax
from pyfunt.padding import Padding
from pyfunt.identity import Identity
from pyfunt.concat_table import ConcatTable
from pyfunt.c_add_table import CAddTable


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


def resnet(n_size, num_starting_filters, reg):
    '''
    Implementation of ["Deep Residual Learning for Image Recognition",Kaiming \
    He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - http://arxiv.org/abs/1512.03385

    Inspired by https://github.com/gcr/torch-residual-networks

    This network should model a similiar behaviour of gcr's implementation.
    Check https://github.com/gcr/torch-residual-networks for more infos about \
    the structure.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    The network has, like in the reference paper (except for the final optional
    affine layers), (6*n)+2 layers, composed as below:

                                                (image_dim: 3, 32, 32; F=16)
                                                (input_dim: N, *image_dim)
         INPUT
            |
            v
       +-------------------+
       |conv[F, *image_dim]|                    (out_shape: N, 16, 32, 32)
       +-------------------+
            |
            v
       +-------------------------+
       |n * res_block[F, F, 3, 3]|              (out_shape: N, 16, 32, 32)
       +-------------------------+
            |
            v
       +-------------------------+
       |res_block[2*F, F, 3, 3]  |              (out_shape: N, 32, 16, 16)
       +-------------------------+
            |
            v
       +---------------------------------+
       |(n-1) * res_block[2*F, 2*F, 3, 3]|      (out_shape: N, 32, 16, 16)
       +---------------------------------+
            |
            v
       +-------------------------+
       |res_block[4*F, 2*F, 3, 3]|              (out_shape: N, 64, 8, 8)
       +-------------------------+
            |
            v
       +---------------------------------+
       |(n-1) * res_block[4*F, 4*F, 3, 3]|      (out_shape: N, 64, 8, 8)
       +---------------------------------+
            |
            v
       +-------------+
       |pool[1, 8, 8]|                          (out_shape: N, 64, 1, 1)
       +-------------+
            |
            v
       +- - - - - - - - -+
       |(opt) m * affine |                      (out_shape: N, 64, 1, 1)
       +- - - - - - - - -+
            |
            v
       +-------+
       |softmax|                                (out_shape: N, num_classes)
       +-------+
            |
            v
         OUTPUT

    Every convolution layer has a pad=1 and stride=1, except for the  dimension
    enhancning layers which has a stride of 2 to mantain the computational
    complexity.
    Optionally, there is the possibility of setting m affine layers immediatley
    before the softmax layer by setting the hidden_dims parameter, which should
    be a list of integers representing the numbe of neurons for each affine
    layer.

    Each residual block is composed as below:

                  Input
                     |
             ,-------+-----.
       Downsampling      3x3 convolution+dimensionality reduction
            |               |
            v               v
       Zero-padding      3x3 convolution
            |               |
            `-----( Add )---'
                     |
                  Output

    After every layer, a batch normalization with momentum .1 is applied.

    Weight initialization (check also layers/init.py and layers/README.md):
    - Inizialize the weights and biases for the affine layers in the same
     way of torch's default mode by calling _init_affine_wb that returns a
     tuple (w, b).
    - Inizialize the weights for the conv layers in the same
     way of torch's default mode by calling init_conv_w.
    - Inizialize the weights for the conv layers in the same
     way of kaiming's mode by calling init_conv_w_kaiming
     (http://arxiv.org/abs/1502.01852 and
      http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-\
      initialization)
    - Initialize batch normalization layer's weights like torch's default by
    calling init_bn_w
    - Initialize batch normalization layer's weights like cgr's first resblock\
    's bn (https://github.com/gcr/torch-residual-networks/blob/master/residual\
           -layers.lua#L57-L59) by calling init_bn_w_gcr.

    num_filters=[16, 16, 32, 32, 64, 64],
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_starting_filters: Number of filters for the first convolution
        layer.
        - n_size: nSize for the residual network like in the reference paper
        - hidden_dims: Optional list number of units to use in the
        fully-connected hidden layers between the fianl pool and the sofmatx
        layer.
        - num_classes: Number of scores to produce from the final affine layer.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
    '''

    nfs = num_starting_filters
    model = Sequential()
    add = model.add
    add(SpatialConvolution(3, nfs, 3, 3, 1, 1, 1, 1))
    add(SpatialBatchNormalization(nfs))
    add(ReLU())

    for i in xrange(1, n_size):
        add(residual_layer(nfs))
    add(residual_layer(nfs, 2*nfs, 2))

    for i in xrange(1, n_size-1):
        add(residual_layer(2*nfs))
    add(residual_layer(2*nfs, 4*nfs, 2))

    for i in xrange(1, n_size-1):
        add(residual_layer(4*nfs))

    add(SpatialAveragePooling(8, 8))
    add(Reshape(nfs*4))
    add(Linear(nfs*4, 10))
    add(LogSoftMax())
    return model
