import numpy as np


def init_conv_w(shape):
    '''
    Initialize convolution layer's weights like torch's linear default
    behaviour, called by _init_conv_weights
    For more infos:
    https://github.com/NVIDIA/DIGITS/blob/master/examples/weight-init/READ\
    ME.md
    https://github.com/torch/nn/blob/master/SpatialConvolution.lua#L31
    '''
    input_n = np.prod(shape[1:])
    std = 1./np.sqrt(input_n)
    return np.random.normal(0, std, shape)


def init_conv_w_kaiming(shape, gain=2.):
    '''
    Initialize convolution layer's weights like torch's nninit kaiming, \
    called by _init_conv_weights
    with gain equal to 'relu' (mapped to the value 2)
    For more infos:
    https://github.com/Kaixhin/nninit
    http://arxiv.org/abs/1502.01852
    http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavie\
    r-initialization
    '''
    input_n = np.prod(shape[1:])
    std = np.sqrt(gain/float(input_n))
    return np.random.normal(0., std, shape)


def init_affine_wb(shape):
    '''
    Initialize affine layer's weights and biases like torch's linear \
    default behaviour, called by _init_affine_weights
    For more infos: https://github.com/torch/nn/blob/master/Linear.lua
    '''
    std = 1./np.sqrt(shape[0])
    w = np.random.normal(0, 3e-4, shape)
    b = np.zeros(shape[1])
    return w, b


def init_affine_wb_th(shape):
    '''
    Initialize affine layer's weights and biases like torch's linear \
    default behaviour, called by _init_affine_weights
    For more infos: https://github.com/torch/nn/blob/master/Linear.lua
    '''
    std = 1./np.sqrt(shape[0])
    w = np.random.uniform(-std, std, shape)
    b = np.random.uniform(-std, std, shape[1])
    return w, b


def init_bn_w(n_ch):
    '''
    Initialize batch nokeyrmalization layer's weights like torch's default
    mode, for more infos:
    https://github.com/torch/nn/blob/master/BatchNormalization.lua
    '''
    return np.random.normal(.6, 0.005, size=n_ch)


def init_bn_w_disp(n_ch):
    '''
    Initialize batch nokeyrmalization layer's weights like torch's default
    mode, for more infos:
    https://github.com/torch/nn/blob/master/BatchNormalization.lua
    '''
    return np.random.normal(.7, .04, size=n_ch)


def init_bn_w_gcr(n_ch):
    '''
    Initialize batch normalization layer's weights like torch's default
    mode, for more infos:
    https://github.com/torch/nn/blob/master/BatchNormalization.lua
    '''
    return np.random.normal(1., 2e-3, n_ch)
