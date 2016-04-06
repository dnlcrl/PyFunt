import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise


def random_flips(X):
    '''
    Take random x-y flips of images.

    Input:
    - X: (N, C, H, W) array of image data.

    Output:
    - An array of the same shape as X, containing a copy of the data in X,
      but with half the examples flipped along the horizontal direction.
    '''

    N, C, H, W = X.shape
    mask = np.random.randint(2, size=N)
    out = np.zeros_like(X)
    out[mask == 1] = X[mask == 1, :, :, ::-1]
    out[mask == 0] = X[mask == 0]
    return out


def add_pad(X, pad):
    '''
    Take random crops of images. For each input image we will generate a random
    crop of that image of the specified size.

    Input:
    - X: (N, C, H, W) array of image data
    - pad: Number of white pixels to add on each side of each image

    Output:
    - Array of shape (N, C, H + 2 * pad, WW + 2 * pad)
    '''
    N, C, H, W = X.shape
    assert pad > 0

    out = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    return out


def random_crops(X, crop_shape, pad=0):
    '''
    Take random crops of images. For each input image we will generate a random
    crop of that image of the specified size.

    Input:
    - X: (N, C, H, W) array of image data
    - crop_shape: Tuple (HH, WW) to which each image will be cropped.

    Output:
    - Array of shape (N, C, HH, WW)
    '''
    if pad:
        X = add_pad(X, pad)
    N, C, H, W = X.shape
    HH, WW = crop_shape
    assert HH < H and WW < W

    out = np.zeros((N, C, HH, WW), dtype=X.dtype)

    np.random.randint((H-HH), size=N)
    y_start = np.random.randint((H-HH), size=N)
    x_start = np.random.randint((W-WW), size=N)

    for i in xrange(N):
        out[i] = X[i, :, y_start[i]:y_start[i]+HH, x_start[i]:x_start[i]+WW]

    return out


def random_rotate(X, max_angle=10):
    N, C, H, W = X.shape
    out = np.zeros_like(X)
    high = np.abs(max_angle) + 1
    low = - np.abs(max_angle)
    for i, x in enumerate(X):
        t = x.transpose(1, 2, 0)
        t = rotate(t, np.random.randint(low, high), resize=False)
        t = t.transpose(2, 0, 1)

        out[i] = t
    return out


def random_contrast(X, scale=(0.8, 1.2)):
    '''
    Randomly adjust the contrast of images. For each input image, choose a
    number uniformly at random from the range given by the scale parameter,
    and multiply each pixel of the image by that number.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: (N, C, H, W) array of image data
    - scale: Tuple (low, high). For each image we sample a scalar in the
      range (low, high) and multiply the image by that scaler.

    Output:
    - Rescaled array out of shape (N, C, H, W) where out[i] is a contrast
      adjusted version of X[i].
    '''
    low, high = scale
    N = X.shape[0]
    out = np.zeros_like(X)
    l = (scale[1]-scale[0])*np.random.random_sample(N)+scale[0]
    # for i in xrange(N):
    #   out[i] = X[i] * l[i]
    out = X * l[:, None, None, None]
    # TODO: vectorize this somehow...
    # out =   #np.diag(l).dot(X)#X*l[:,np.newaxis, np.newaxis, np.newaxis]

    return out


def random_tint(X, scale=(-10, 10)):
    '''
    Randomly tint images. For each input image, choose a random color whose
    red, green, and blue components are each drawn uniformly at random from
    the range given by scale. Add that color to each pixel of the image.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: (N, C, W, H) array of image data
    - scale: A tuple (low, high) giving the bounds for the random color that
      will be generated for each image.

    Output:
    - Tinted array out of shape (N, C, H, W) where out[i] is a tinted version
      of X[i].
    '''
    low, high = scale
    N, C = X.shape[:2]
    out = np.zeros_like(X)

    l = (scale[1]-scale[0])*np.random.random_sample((N, C))+scale[0]
    out = X+l[:, :, None, None]

    return out


def fixed_crops(X, crop_shape, crop_type):
    '''
    Take center or corner crops of images.
    source:
    https://github.com/MyHumbleSelf/cs231n/blob/master/assignment3/cs231n/data_augmentation.py

    Inputs:
    - X: Input data, of shape (N, C, H, W)
    - crop_shape: Tuple of integers (HH, WW) giving the size to which each
      image will be cropped.
    - crop_type: One of the following strings, giving the type of crop to
      compute:
      'center': Center crop
      'ul': Upper left corner
      'ur': Upper right corner
      'bl': Bottom left corner
      'br': Bottom right corner

    Returns:
    Array of cropped data of shape (N, C, HH, WW)
    '''
    N, C, H, W = X.shape
    HH, WW = crop_shape

    x0 = (W - WW) / 2
    y0 = (H - HH) / 2
    x1 = x0 + WW
    y1 = y0 + HH

    if crop_type == 'center':
        return X[:, :, y0:y1, x0:x1]
    elif crop_type == 'ul':
        return X[:, :, :HH, :WW]
    elif crop_type == 'ur':
        return X[:, :, :HH, -WW:]
    elif crop_type == 'bl':
        return X[:, :, -HH:, :WW]
    elif crop_type == 'br':
        return X[:, :, -HH:, -WW:]
    else:
        raise ValueError('Unrecognized crop type %s' % crop_type)


def RGB_PCA(images):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py

    RGB PCA and variations from Alex's paper
     '''
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels)/256.
    C = np.cov(pixels)/(256.*256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py
     '''
    a = np.random.randn(3)
    v = np.array([a[0]*eig_val[0], a[1]*eig_val[1], a[2]*eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


def noise(x):
    '''
    Source: https://github.com/Thrandis/ift6266h15/blob/1cc3fc6164dc6c54936971
    935027cd447e2cd81f/dataset_augmentation.py
     '''
    r = np.random.rand(1)[0]
    # TODO randomize parameters of the noises; check how to init seed
    if r < 0.33:
        return random_noise(x, 's&p', seed=np.random.randint(1e6))
    if r < 0.66:
        return random_noise(x, 'gaussian', seed=np.random.randint(1e6))
    return random_noise(x, 'speckle', seed=np.random.randint(1e6))
