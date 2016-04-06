import numpy as np
import math


def relu_forward(x, in_place=True):
    '''
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Threshold is set 1e-6 to avoid potential divdes by zero:
    https://github.com/torch/nn/commit/ab09f77b32119e0c2de49572c8c856c81363c2a0
    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    '''
    th = 1e-6
    out = x
    indices = np.where(x > th)
    cache = indices, in_place
    if in_place:
        x[indices] = 0.
        return x, cache

    out[indices] = 0.
    return out, cache


def relu_backward(dout, cache):
    '''
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    '''
    indices, in_place = cache
    if in_place:
        dout[indices] = 0.
        return dout
    dx = dout
    dx[indices] = 0.
    return dx


def affine_forward(x, w, b):
    '''
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M)
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    '''
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w)
    return out, cache


def affine_backward(dout, cache):
    '''
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    '''
    x, w = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def batchnorm_forward(x, gamma, beta, bn_param):
    '''
    Forward pass for batch normalization like Torch:
    https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    '''
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', .1)

    N, D = x.shape
    running_mean = bn_param.get(
        'running_mean', np.zeros(D, dtype=np.float64))  # dtype=x.dtype
    running_var = bn_param.get('running_var', np.ones(D, dtype=np.float64))

    out, cache = None, None
    if mode == 'train':
        mean = 1. / N * np.sum(x, axis=0)

        xmu = x - mean

        carre = xmu*xmu

        var = 1. / N * np.sum(carre, axis=0)

        sqrtvar = np.sqrt(var + eps)

        invstd = 1. / sqrtvar

        running_mean = momentum * mean + (1. - momentum) * running_mean

        unbiased_var = np.sum(carre, axis=0)/(N - 1.)

        running_var = momentum * unbiased_var + (1. - momentum) * running_var

        cache = (xmu, invstd, gamma)

    elif mode == 'test':
        mean = running_mean
        invstd = 1. / np.sqrt(running_var + eps)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    out = ((x - mean) * invstd) * gamma + beta
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = np.array(running_mean, copy=True)
    bn_param['running_var'] = np.array(running_var, copy=True)

    return out, cache


def batchnorm_backward(dout, cache):
    '''
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    '''
    xmu, invstd, gamma = cache

    N, D = dout.shape

    _sum = np.sum(dout, axis=0)
    dotp = np.sum((xmu * dout), axis=0)

    k = 1. / N * dotp * invstd * invstd
    dx = xmu * k

    dmean = 1. / N * _sum
    dx = (dout - dmean - dx) * invstd * gamma

    dgamma = dotp * invstd

    dbeta = _sum

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    '''
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    '''
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    x_flat = np.ascontiguousarray(x_flat, dtype=x.dtype)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    '''
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    '''
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dout_flat = np.ascontiguousarray(dout_flat, dtype=dout_flat.dtype)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    '''
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    '''
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        # Inverted dropout mask. Notice /p!
        # normilizing by p leads to multiplicative factor
        # this mask is sufficient since it's already binary
        mask = (np.random.rand(*x.shape) < p)

        out = x*mask  # drop!
    elif mode == 'test':
        # Do nothing!
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    '''
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    '''
    dropout_param, mask = cache
    mode = dropout_param['mode']
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def svm_loss(x, y):
    '''
    Computes the loss and gradient for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    '''
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def log_softmax_loss(x, y):
    '''
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    '''
    N = x.shape[0]
    xdev = x - x.max(1, keepdims=True)
    logp = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
    loss = -np.mean(logp[np.arange(N), y])

    dx = np.exp(logp)
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    '''
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    '''
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def max_pool_forward_naive(x, pool_param):
    '''
    A naive implementation of the forward pass for a max pooling layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    '''

    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) / stride
    Wp = 1 + (W - WW) / stride

    out = np.zeros((N, C, Hp, Wp))

    for i in xrange(N):
        # Need this; apparently we are required to max separately over each
        # channel
        for j in xrange(C):
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window (C, HH, WW)
                    window = x[i, j, hs:hs+HH, ws:ws+WW]
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    '''
    A naive implementation of the backward pass for a max pooling layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    '''

    x, pool_param = cache
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) / stride
    Wp = 1 + (W - WW) / stride

    dx = np.zeros_like(x)

    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window (C, HH, WW)
                    window = x[i, j, hs:hs+HH, ws:ws+WW]
                    m = np.max(window)

                    # Gradient of max is indicator
                    dx[i, j, hs:hs+HH, ws:ws +
                        WW] += (window == m) * dout[i, j, k, l]

    return dx


def avg_pool_forward_naive(x, pool_param):
    '''
    A naive implementation of the forward pass for a avg pooling layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    '''

    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) / stride
    Wp = 1 + (W - WW) / stride

    out = np.zeros((N, C, Hp, Wp))

    for i in xrange(N):
        # Need this; apparently we are required to max separately over each
        # channel
        for j in xrange(C):
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window (C, HH, WW)
                    window = x[i, j, hs:hs+HH, ws:ws+WW]
                    out[i, j, k, l] = np.mean(window)

    cache = (x, pool_param)
    return out, cache


def avg_pool_backward_naive(dout, cache):
    '''
    A naive implementation of the backward pass for a avg pooling layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    '''

    x, pool_param = cache
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) / stride
    Wp = 1 + (W - WW) / stride

    dx = np.zeros_like(x)

    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window (C, HH, WW)
                    window = x[i, j, hs:hs+HH, ws:ws+WW]
                    n = window.size

                    # Gradient of mean is 1/n
                    dx[i, j, hs:hs+HH, ws:ws +
                        WW] += dout[i, j, k, l] / np.float(n)

    return dx


def conv_forward_naive(x, w, b, conv_param):
    '''
    A naive implementation of the forward pass for a convolutional layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    '''
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = int(math.floor((H + 2 * pad - HH) / stride + 1))
    Wp = int(math.floor((W + 2 * pad - WW) / stride + 1))

    out = np.zeros((N, F, Hp, Wp))

    # Add padding around each 2D image
    padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    for i in xrange(N):  # ith example
        for j in xrange(F):  # jth filter

            # Convolve this filter over windows
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window we want to apply the respective jth filter over
                    # (C, HH, WW)
                    window = padded[i, :, hs:hs+HH, ws:ws+WW]

                    # Convolve
                    out[i, j, k, l] = np.sum(window*w[j]) + b[j]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    '''
    A naive implementation of the backward pass for a convolutional layer.
    source: https://github.com/MyHumbleSelf/cs231n/blob/master/assignment2/cs231n/layers.py

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    '''
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = int(math.floor((H + 2 * pad - HH) / stride + 1))
    Wp = int(math.floor((W + 2 * pad - WW) / stride + 1))

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Add padding around each 2D image (and respective gradient)
    # There may be a prettier way to do this but I can't think of any nice way at
    # least. You want to contribute to the boundary sums and in some cases the
    # only way to do that is by writing into the padding. I'm sure some very nasty
    # indexing trick will do; with lots of floors and ceils.
    padded = np.pad(
        x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    padded_dx = np.pad(
        dx, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    for i in xrange(N):  # ith example
        for j in xrange(F):  # jth filter
            # Convolve this filter over windows
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window we applies the respective jth filter over (C, HH,
                    # WW)
                    window = padded[i, :, hs:hs+HH, ws:ws+WW]

                    # Compute gradient of out[i, j, k, l] = np.sum(window*w[j])
                    # + b[j]
                    db[j] += dout[i, j, k, l]
                    dw[j] += window*dout[i, j, k, l]
                    padded_dx[i, :, hs:hs+HH, ws:ws+WW] += w[j] * \
                        dout[i, j, k, l]

    # "Unpad"
    dx = padded_dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db
