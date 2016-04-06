# ./nnet/data/ Directory

In this folder you will find everything you need to load and use yor datasets.
You should also use this directory as root for your datasets.

## Directory Structure
	.
	+-- __init__.py
	+-- data_augmentation.py
	+-- data_utils.py
	+-- get_datasets.sh
	+-- gtsrb.py
	+-- README.md

## data_augmentation.py

Data augmentation utilities, contains various functions to augment your datasets.

## data_utils.py

Data loading utilities, contains functions to load the CIFAR10 and Tiny Imagenet datasets and trained models files.

## get_datasets.sh

Download and expand the CIFAR10 dataset. Run this script before running train.py

## gtsrb.py

Functions to load the GTSRB dataset.



