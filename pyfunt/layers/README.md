# ./nnet/layers Directory

In this folder you will find everything you need to load and use yor datasets.

## Directory Structure
	.
	+-- __init__.py
	+-- fast_layers.py
	+-- im2col.py
	+-- im2col_cython.pyx
	+-- init.py
	+-- layer_utils.py
	+-- layers.py
	+-- README.md
	+-- setup.py

## fast_layers.py

Fast implementations of forward/backward pass for:

- convolution layers:
	- conv_forward_im2col, conv_backward_im2col:
		A fast implementation of the forward pass for a convolutional layer
		based on im2col and col2im.
	- conv_forward_strides, conv_backward_strides:
		A faster implementation of the forward pass for a convolutional layers based on striding.
- max pooling layers:
	- max_pool_forward_fast, max_pool_backward_fast:
		Chose between im2col and reshape methods based on the input shape and pooling params.
	- max_pool_forward_im2col, max_pool_backward_im2col:
		A fast implementation of the forward pass for a max pool layer
		based on im2col and col2im.
	- max_pool_forward_reshape, max_pool_backward_reshape:
		A faster implementation of the forward pass for a max pool layer
		based on reshaping and broadcasting.
- average pooling layers:
	- avg_pool_forward_fast, avg_pool_backward_fast:
		Chose between im2col and reshape methods based on the input shape and pooling params.
	- avg_pool_forward_im2col, max_pool_backward_im2col:
		A fast implementation of the forward pass for a avg pool layer
		based on im2col and col2im.
	- avg_pool_forward_reshape, max_pool_backward_reshape:
		A faster implementation of the forward pass for a avg pool layer
		based on reshaping and broadcasting.

## im2col.py

Helper functions to use is fast_layers.py: check [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/) for more infos

## im2col_cython.pyx

Helper functions to use is fast_layers.py: check [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/) for more infos

## init.py

Weight initialization:

- Inizialize the weights and biases for the affine layers in the same way of torch's default mode by calling _init_affine_wb that returns a tuple (w, b).
- Inizialize the weights for the conv layers in the same way of torch's default mode by calling init_conv_w.
- Inizialize the weights for the conv layers in the same way of kaiming's mode by calling init_conv_w_kaiming ([http://arxiv.org/abs/1502.01852](http://arxiv.org/abs/1502.01852) and [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization))
- Initialize batch normalization layer's weights like torch's default by calling init_bn_w
- Initialize batch normalization layer's weights like cgr's first resblock's bn ([https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua#L57-L59](https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua#L57-L59)) by calling init_bn_w_gcr.

## layer_utils.py

Convenience layers definitions:

- skip:
	a skip path for residual networks ([http://arxiv.org/abs/1512.03385](http://arxiv.org/abs/1512.03385)), eventually appling pad and pooling if the number of channels is doubled (by the convolution layers).
- affine_relu:
	an affine transform followed by a ReLU.
- affine_batchnorm_relu:
	an affine transform followed by batch normalization, followed by a ReLU.
- conv_relu:
	a convolution followed by a ReLU.
- conv_relu_pool:
	a convolution, a ReLU, and a pool.
- conv_batchnorm_relu:
	a convolution followed by a batch normalization, followed by a ReLU.
- conv_batchnorm_relu_pool:
	a convolution followed by a batch normalization, followed by a ReLU, followed by a pooling layer

## layers.py

Layers definitions:

- relu:
	layer of rectified linear units (ReLUs), just like torch's implementation ([https://github.com/torch/nn/blob/master/Threshold.lua](https://github.com/torch/nn/blob/master/Threshold.lua));
- affine:
	affine (fully-connected) layer;
- batchnorm:
	batch normalization like torch's .c implementation ([https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c)), applyable on input of shape (M, C);
- spatial_batchnorm:
	batch normalize an input of shape (N, C, H, W);
- dropout:
	(inverted) dropout (check [http://cs231n.github.io/neural-networks-2/](http://cs231n.github.io/neural-networks-2/) for more infos)
- svm_loss:
	computes the loss and gradient for multiclass SVM classification.
- log_softmax_loss:
	computes the logsoftmax loss and gradient like torch's LogSoftMAx + ClassNLLCriterion ([https://github.com/torch/nn/blob/master/doc/criterion.md](https://github.com/torch/nn/blob/master/doc/criterion.md), [https://github.com/torch/nn/blob/master/lib/THNN/generic/LogSoftMax.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/LogSoftMax.c) and [https://github.com/torch/nn/blob/master/lib/THNN/generic/ClassNLLCriterion.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/ClassNLLCriterion.c));
- softmax_loss:
	computes the softmax loss and gradient. This is the default implementations for the cs231n class;

## setup.py

File neede to compile the im2col_cython.pyx so it can produce the .c extension file to load in fast_layers.py. to build the needed im2col_cython.c run:
	
	python setup.py build_ext --inplace



