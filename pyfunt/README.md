# ./nnet/ Directory

In this folder contains solutions for cs231n assignments and some other cool staff.

	# Directory Structure
	.
	+-- __init__.py
	+-- optim.py
	+-- solver.py
	+-- utils/
	+-- layers/
	+-- data/

## optim.py

Contains various first-order update rules that are commonly used for training neural networks:

- sgd_th:
	Stochastic gradient descent with nesterov momentum, like Torch's optim.sgd:
	https://github.com/torch/optim/blob/master/sgd.lua
- nesterov:
	stochastic gradient descent with nesterov momentum.
- sgd_momentum:
	stochastic gradient descent with momentum.
- sgd:
	standard stochastic gradient descent.
- rmsprop:
	uses the RMSProp update rule, which uses a moving average of squared gradient values to set adaptive per-parameter learning rates.
- adam:
	adam update rule, which incorporates moving averages of both the
	gradient and its square and a bias correction term.

## solver.py

Contains the Solver class, which is the trainer for your networks, plus some function to permit training with multiple processes.

## utils/

Check utils/README.md

## layers/

Check layers/README.md

## data/

Check data/README.md

