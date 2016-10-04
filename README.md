# PyFunt (/paɪfʊnt/)

Pythonic Deep Learning Framework (WIP and CPU only at the moment) inspired by [Torch](http://torch.ch)'s [Neural Network package](https://github.com/torch/nn)

## Requirements

- [Python 2.7](https://www.python.org/)
- [Cython](cython.org/)
- [numpy](www.numpy.org/)
- [scikit_learn](scikit-learn.org/)


## Installation

Get [pip](https://pypi.python.org/pypi/pip) and run:

	pip install git+git://github.com/dnlcrl/PyFunt.git

## Usage

Check the [examples folder](https://github.com/dnlcrl/PyFunt/tree/master/pyfunt/examples)

### Example: Parametric Residual Model

Parametric models can be built easily thanks to the module structure:

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
	    res.add(ReLU(True))
	    return res


	def resnet(n_size, num_starting_filters, reg):
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

---
 
Check the Torch documentation for more informations about the implemented layers (pyfunt is more or less a python port of torch/nn): [https://github.com/torch/nn/blob/master/doc/index.md](https://github.com/torch/nn/blob/master/doc/index.md)
