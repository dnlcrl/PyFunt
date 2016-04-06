# PyFunt

### Pythonic Deep Learning Framework

## Description 

This repo mostly contains my solutions for Stanford's cs231n assignments, extended and eventually readapted to be part of a single framework, or to bring some feature from other framework like Torch, plus some other cool staff.

## Requirements

- [Python 2.7](https://www.python.org/)
- [Cython](cython.org/)
- [matplotlib](matplotlib.org/)
- [numpy](www.numpy.org/)
- [scipy](www.scipy.org/)
- [cv2](opencv.org) (only for loading GTSRB)
- [scikit_learn](scikit-learn.org/)

After you get Python, you can get [pip](https://pypi.python.org/pypi/pip) and install all requirements by running:
	
	pip install -r /path/to/requirements.txt


# Directory Structure
.
+-- README.md
+-- __init__.py
+-- optim.py
+-- solver.py
+-- utils/
+-- layers/
+-- data/

## README.md

This File.

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


