import numpy as np

'''
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
'''


def sgd_th(w, dw, config=None):
    '''
    Performs stochastic gradient descent with nesterov momentum,
    like Torch's optim.sgd:
    https://github.com/torch/optim/blob/master/sgd.lua

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - nesterov: Boolean to indicate if nesterov momentum should be applied
    - dampening: default equal to momentum.
    - weight_decay: apply weight_decay in place.
    - state_dw: stored gradients for the next update.
    '''
    if config is None:
        config = {}

    learning_rate = config.get('learning_rate', 1e-2)
    momentum = config.get('momentum', 0)
    nesterov = config.get('nesterov', False)
    dampening = config.get('dampening', 0)
    weight_decay = config.get('weight_decay', 0)
    state_dw = config.get('state_dw', None)
    assert (not nesterov or (momentum > 0 and dampening == 0)
            ), 'Nesterov momentum requires a momentum and zero dampening'
    dampening = dampening or momentum
    dw = dw.copy()
    if weight_decay:
        dw += weight_decay * w

    if momentum:
        if state_dw is None:
            state_dw = dw
        else:
            state_dw *= momentum
            state_dw += (1 - dampening) * dw
        if nesterov:
            dw = dw + momentum * state_dw
        else:
            dw = state_dw

    next_w = w - learning_rate * dw

    config['state_dw'] = state_dw

    return next_w, config


def nesterov(w, dw, config=None):
    '''
    Performs stochastic gradient descent with nesterov momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    '''
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w, dtype=np.float64))

    next_w = None
    prev_v = v
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w - config['momentum'] * prev_v + (1 + config['momentum']) * v
    config['velocity'] = v

    return next_w, config


def sgd(w, dw, config=None, p=-1):
    '''
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    '''
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    '''
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    '''
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v + config['learning_rate'] * dw
    next_w = w - v
    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    '''
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    '''
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    cache = config['cache']
    decay_rate = config['decay_rate']
    learning_rate = config['learning_rate']
    cache = decay_rate * cache + (1 - decay_rate) * dx**2
    x += - learning_rate * dx / (np.sqrt(cache) + 1e-8)

    config['cache'] = cache
    next_x = x

    return next_x, config


def adam(x, dx, config=None):
    '''
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    '''
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    m = config['m']
    v = config['v']
    t = config['t']
    beta1 = config['beta1']
    beta2 = config['beta2']

    # update parameters
    learning_rate = config['learning_rate']
    epsilon = config['epsilon']
    m = beta1*m + (1-beta1)*dx
    v = beta2*v + (1-beta2)*dx**2
    t = t + 1
    next_x = x - learning_rate*m/(np.sqrt(v) + epsilon)

    # Writing back in config
    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_x, config
