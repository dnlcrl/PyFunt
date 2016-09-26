import numpy as np
from spatial_convolution import SpatialConvolution
from spatial_batch_normalitazion import SpatialBatchNormalization
from spatial_average_pooling import SpatialAveragePooling
from sequential import Sequential
from padding import Padding
from relu import ReLU
from c_add_table import CAddTable
from residual_layer import add_residual_layer
from linear import Linear


def ResNet(N):

    model = Sequential()
    add = model.add
    add(SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
    add(SpatialBatchNormalization(16))
    add(ReLU(True))

    for i in xrange(1, N):
        add_residual_layer(16)
    add_residual_layer(16, 32, 2)

    for i in xrange(1, N-1):
        add_residual_layer(32)
    add_residual_layer(32, 64, 2)

    for i in xrange(1, N-1):
        add_residual_layer(64)

    add(SpatialAveragePooling(8, 8))
    add(Reshape(64))
    add(Linear(64, 10))
    add(LogSoftMax())


class ResNet0(object):

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
    '''

    def __init__(self, n_size=1, input_dim=(3, 32, 32), num_starting_filters=16,
                 hidden_dims=[], num_classes=10, reg=0.0, weights=None, dtype=np.float32):
        '''
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
        self.params={}
        self.reg=reg
        self.dtype=dtype
        self.bn_params={}
        self.n_size=n_size
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.filter_size=3

        self.num_filters=self._init_filters(num_starting_filters)
        self.conv_layers=len(self.num_filters)
        self.affine_layers=len(hidden_dims)
        self.pool_l=self.conv_layers + 1
        self.softmax_l=self.conv_layers + self.affine_layers + 2
        self.affine_l=self.conv_layers + 2
        self.return_probs=False

        self._init_conv_weights()

        assert (self.input_dim[-1] % 4) == 0
        p=self.input_dim[-1] / 4
        self.final_pool_param={
            'stride': p, 'pool_height': p, 'pool_width': p}

        self.h_dims=(
            [hidden_dims[0]] if len(hidden_dims) > 0 else []) + hidden_dims

        if weights:
            for k, v in weights.iteritems():
                self.params[k]=v.astype(dtype)
            return

        self._init_affine_weights()

        self._init_scoring_layer(num_classes)

        for k, v in self.params.iteritems():
            self.params[k]=v.astype(dtype)

    def __str__(self):
        return """
        Residual Network:
            NSize: %d;
            Numbers of filters for each layer:%s;
            Optional linear layers dimensions: %s;
            Regularization factor: %f;
        """ % (self.n_size,
               str(self.num_filters),
               str(self.hidden_dims),
               self.reg)

    def _init_filters(self, nf):
        '''
        Initialize conv filters like in
        https://github.com/gcr/torch-residual-networks
        Called by self.__init__
        '''

        num_filters=[nf]  # first conv
        for i in range(self.n_size):  # n res blocks
            num_filters += [nf] * 2
        nf *= 2
        num_filters += [nf] * 2  # res block increase ch
        for i in range(self.n_size-1):  # n-1 res blocks
            num_filters += [nf] * 2

        nf *= 2
        num_filters += [nf] * 2  # res block increase ch
        for i in range(self.n_size-1):  # n-1 res blocks
            num_filters += [nf] * 2
        return num_filters

    def _init_conv_weights(self):
        '''
        Initialize conv weights.
        Called by self.__init__
        '''
        # Size of the input
        Cinput, Hinput, Winput=self.input_dim
        filter_size=self.filter_size

        # Initialize the weight for the conv layers
        F=[Cinput] + self.num_filters
        for i in xrange(self.conv_layers):
            idx=i + 1
            shape=F[i + 1], F[i], filter_size, filter_size
            out_ch=shape[0]
            W=init_conv_w_kaiming(shape)
            b=np.zeros(out_ch)
            self.params['W%d' % idx]=W
            self.params['b%d' % idx]=b
            bn_param={'mode': 'train',
                        'running_mean': np.zeros(out_ch),
                        'running_var': np.ones(out_ch)}
            # if i % 2 == 0 else init_bn_w_disp(out_ch)
            gamma=init_bn_w(out_ch)
            beta=np.zeros(out_ch)
            self.bn_params['bn_param%d' % idx]=bn_param
            self.params['gamma%d' % idx]=gamma
            self.params['beta%d' % idx]=beta

        # Initialize conv/pools parameters
        self.conv_param1={'stride': 1, 'pad': (filter_size - 1) / 2}
        self.conv_param2={'stride': 2, 'pad': 0}
        self.pool_param1={'pool_height': 2, 'pool_width': 2, 'stride': 2}

    def _init_affine_weights(self):
        '''
        Initialize affine weights.
        Called by self.__init__
        '''
        dims=self.h_dims
        for i in xrange(self.affine_layers):
            idx=self.affine_l + i
            shape=dims[i], dims[i + 1]
            out_ch=shape[1]
            W, b=init_affine_wb(shape)
            self.params['W%d' % idx]=W
            self.params['b%d' % idx]=b
            bn_param={'mode': 'train',
                        'running_mean': np.zeros(out_ch),
                        'running_var': np.ones(out_ch)}
            gamma=np.ones(out_ch)
            beta=np.zeros(out_ch)
            self.bn_params['bn_param%d' % idx]=bn_param
            self.params['gamma%d' % idx]=gamma
            self.params['beta%d' % idx]=beta

    def _init_scoring_layer(self, num_classes):
        '''
        Initialize scoring layer weights.
        Called by self.__init__
        '''
        # Scoring layer
        in_ch=self.h_dims[-1] if \
            len(self.h_dims) > 0 else self.num_filters[-1]
        shape=in_ch, num_classes
        W, b=init_affine_wb(shape)
        i=self.softmax_l
        self.params['W%d' % i]=W
        self.params['b%d' % i]=b

    def _extract(self, params, idx, bn=True):
        '''
        Ectract Parameters from params
        '''
        w=params['W%d' % idx]
        b=params['b%d' % idx]
        if bn:
            beta=params['beta%d' % idx]
            gamma=params['gamma%d' % idx]
            bn_param=self.bn_params['bn_param%d' % idx]
            return w, b, beta, gamma, bn_param
        return w, b

    def _put(self, cache, idx, h, cache_h):
        '''
        Put h and h_cache in cache
        '''
        cache['h%d' % idx]=h
        cache['cache_h%d' % idx]=cache_h
        return cache

    def _put_grads(self, cache, idx, dh, dw, db, dbeta=None, dgamma=None):
        '''
        Put grads in cache
        '''
        cache['dh%d' % (idx - 1)]=dh
        cache['dW%d' % idx]=dw
        cache['db%d' % idx]=db
        if dbeta is not None:
            cache['dbeta%d' % idx]=dbeta
            cache['dgamma%d' % idx]=dgamma
        return cache

    def _forward_first_conv(self, cache):
        i=0
        idx=i + 1
        h=cache['h%d' % (idx - 1)]
        w, b, beta, gamma, bn_param=self._extract(self.params, idx)

        conv_param=self.conv_param1

        c_out, conv_cache=conv_forward(h, w, b, conv_param)
        bn_out, batchnorm_cache=spatial_batchnorm_forward(
            c_out, gamma, beta, bn_param)
        h, relu_cache=relu_forward(bn_out)
        cache_h=(conv_cache, batchnorm_cache, relu_cache)

        self._put(cache, idx, h, cache_h)

    def _forward_convs(self, cache):
        '''
        Execute convolution layers' forward pass
        '''
        for i in xrange(1, self.conv_layers):
            idx=i + 1
            h=cache['h%d' % (idx - 1)]
            w, b, beta, gamma, bn_param=self._extract(self.params, idx)

            out_ch, in_ch=w.shape[:2]
            if i > 0 and i % 2 == 1:
                # store skip
                skip, skip_cache=skip_forward(h, out_ch)
                cache['cache_skip%d' % idx]=skip_cache
            if i == 0 or in_ch == out_ch:
                conv_param=self.conv_param1
            else:
                conv_param=self.conv_param2
                h=np.pad(
                    h, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')

            c_out, conv_cache=conv_forward(h, w, b, conv_param)
            bn_out, batchnorm_cache=spatial_batchnorm_forward(
                c_out, gamma, beta, bn_param)
            h, relu_cache=relu_forward(bn_out)
            cache_h=(conv_cache, batchnorm_cache, relu_cache)

            if i > 0 and i % 2 == 0:
                # add skip
                h += skip
            self._put(cache, idx, h, cache_h)

    def _forward_affines(self, cache):
        '''
        Execute affine layers's forward pass
        '''
        for i in xrange(self.affine_layers):
            idx=self.affine_l + i
            h=cache['h%d' % (idx - 1)]
            w, b, beta, gamma, bn_param=self._extract(self.params, idx)

            h, cache_h=affine_bn_relu_forward(h, w, b, gamma,
                                                beta, bn_param)
            self._put(cache, idx, h, cache_h)

    def _forward_pool(self, cache):
        '''
        Execute pooling layer's forward pass
        '''
        idx=self.pool_l
        h=cache['h%d' % (idx - 1)]
        h, cache_h=avg_pool_forward(h, self.final_pool_param)
        self._put(cache, idx, h, cache_h)

    def _forward_score_layer(self, cache):
        '''
        Execute softmax layer's forward pass
        '''
        idx=self.softmax_l
        w, b=self._extract(self.params, idx, bn=False)
        h=cache['h%d' % (idx - 1)]
        h, cache_h=affine_forward(h, w, b)
        self._put(cache, idx, h, cache_h)

    def _backward_score_layer(self, dscores, cache):
        '''
        Execute softmax layer's backward pass
        '''
        idx=self.softmax_l
        dh=dscores
        h_cache=cache['cache_h%d' % idx]
        dh, dw, db=affine_backward(dh, h_cache)
        self._put_grads(cache, idx, dh, dw, db)

    def _backward_pool(self, cache):
        '''
        Execute pooling layer's backward pass
        '''
        idx=self.pool_l
        dh=cache['dh%d' % idx]
        h_cache=cache['cache_h%d' % idx]
        dh=avg_pool_backward(dh, h_cache)
        cache['dh%d' % (idx - 1)]=dh

    def _backward_affines(self, cache):
        '''
        Execute affine layers' backward pass
        '''
        for i in range(self.affine_layers)[::-1]:
            idx=self.affine_l + i
            dh=cache['dh%d' % idx]
            h_cache=cache['cache_h%d' % idx]
            dh, dw, db, dgamma, dbeta=affine_bn_relu_backward(
                dh, h_cache)
            self._put_grads(cache, idx, dh, dw, db, dbeta, dgamma)

    def _backward_convs(self, cache):
        '''
        Execute convolution layers' backward pass
        '''
        for i in range(1, self.conv_layers)[::-1]:
            idx=i + 1
            dh=cache['dh%d' % idx]
            h_cache=cache['cache_h%d' % idx]
            w=self.params['W%d' % idx]

            out_ch, in_ch=w.shape[:2]
            if i > 0 and i % 2 == 0:
                skip_cache=cache['cache_skip' + str(idx-1)]
                dskip=skip_backward(dh, skip_cache)

            if i == self.conv_layers-1:
                dh=dh.reshape(*cache['h%d' % idx].shape)

            conv_cache, batchnorm_cache, relu_cache=h_cache
            dbn=relu_backward(dh, relu_cache)
            dconv, dgamma, dbeta=spatial_batchnorm_backward(
                dbn, batchnorm_cache)
            dh, dw, db=conv_backward(dconv, conv_cache)

            if not(i == 0 or in_ch == out_ch):
                # back pad trick
                dh=dh[:, :, 1:, 1:]

            if i > 0 and i % 2 == 1:
                dh += dskip

            self._put_grads(cache, idx, dh, dw, db, dbeta, dgamma)

    def _backward_first_conv(self, cache):
        '''
        Execute convolution layers' backward pass
        '''
        i=0
        idx=i + 1
        dh=cache['dh%d' % idx]
        h_cache=cache['cache_h%d' % idx]

        if i == self.conv_layers-1:
            dh=dh.reshape(*cache['h%d' % idx].shape)

        conv_cache, batchnorm_cache, relu_cache=h_cache
        dbn=relu_backward(dh, relu_cache)
        dconv, dgamma, dbeta=spatial_batchnorm_backward(dbn, batchnorm_cache)
        dh, dw, db=conv_backward(dconv, conv_cache)

        self._put_grads(cache, idx, dh, dw, db, dbeta, dgamma)

    def loss_helper(self, args):
        '''
        Helper method used to call loss() within a pool of processes using \
        pool.map_async.
        '''
        return self.loss(*args)

    def loss(self, X, y=None, compute_dX=False):
        '''
        Evaluate loss and gradient for the three-layer convolutional network.
        '''
        X=X.astype(self.dtype)
        mode='test' if y is None else 'train'
        params=self.params
        for key, bn_param in self.bn_params.iteritems():
            bn_param[mode]=mode
        scores=None

        cache={}
        cache['h0']=X

        self._forward_first_conv(cache)

        self._forward_convs(cache)

        self._forward_pool(cache)

        self._forward_affines(cache)

        self._forward_score_layer(cache)

        scores=cache['h%d' % self.softmax_l]

        if y is None:
            if self.return_probs:
                probs=np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs /= np.sum(probs, axis=1, keepdims=True)
                return probs

            return scores

        loss, grads=0, {}

        data_loss, dscores=log_softmax_loss(scores, y)

        # Backward pass
        self._backward_score_layer(dscores, cache)

        self._backward_affines(cache)

        self._backward_pool(cache)

        self._backward_convs(cache)

        self._backward_first_conv(cache)

        if compute_dX:
            return cache['dh0']

        # apply regularization to ALL parameters
        grads={}
        reg_loss=.0
        for key, val in cache.iteritems():
            if key[:1] == 'd' and 'h' not in key:  # all params gradients
                reg_term=0
                if self.reg:
                    reg_term=self.reg * params[key[1:]]
                    w=params[key[1:]]
                    reg_loss += 0.5 * self.reg * np.sum(w * w)
                grads[key[1:]]=val + reg_term

        loss=data_loss + reg_loss

        return loss, grads
