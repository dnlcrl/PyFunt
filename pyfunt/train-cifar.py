# !/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import numpy as np
# import matplotlib.pyplot as plt
from pydatset.cifar10 import get_CIFAR10_data
from pydatset.data_augmentation import (random_flips,
                                        random_crops)
from resnet import ResNet
from solver import Solver as Solver

import inspect
import argparse

from class_nll_criterion import ClassNLLCriterion

np.random.seed(0)

DATA_PATH = 'CIFAR_DATASET_PATH'

path_set = False
while not path_set:
    try:
        with open(DATA_PATH) as f:
            DATASET_PATH = f.read()
        path_set = True
    except:
        data_path = raw_input('Enter the path for the CIFAR10 dataset: ')
        with open(DATA_PATH, "w") as f:
            f.write(data_path)


EXPERIMENT_PATH = '../Experiments/' + str(uuid.uuid4())[-10:]

# residual network constants
NSIZE = 3
N_STARTING_FILTERS = 16

# solver constants
NUM_PROCESSES = 4

NUM_TRAIN = 50000
NUM_TEST = 10000

WEIGHT_DEACY = 1e-4
REGULARIZATION = 0
LEARNING_RATE = .1
MOMENTUM = .99
NUM_EPOCHS = 160
BATCH_SIZE = 64
CHECKPOINT_EVERY = 20

XH, XW = 32, 32

args = argparse.Namespace()


def parse_args():
    """
    Parse the options for running the Residual Network on CIFAR-10.
    """
    desc = 'Train a Residual Network on CIFAR-10.'
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--dataset_path',
        metavar='DIRECOTRY',
        default=DATASET_PATH,
        type=str,
        help='directory where results will be saved')
    add('--experiment_path',
        metavar='DIRECOTRY',
        default=EXPERIMENT_PATH,
        type=str,
        help='directory where results will be saved')
    add('-load', '--load_checkpoint',
        metavar='DIRECOTRY',
        default='',
        type=str,
        help='load checkpoint from load_checkpoint')
    add('--n_size',
        metavar='INT',
        default=NSIZE,
        type=int,
        help='Network will have (6*n)+2 conv layers')
    add('--n_starting_filters',
        metavar='INT',
        default=N_STARTING_FILTERS,
        type=int,
        help='Network will starts with those number of filters')
    add('--n_processes', '-np',
        metavar='INT',
        default=NUM_PROCESSES,
        type=int,
        help='Number of processes for each step')
    add('--n_train',
        metavar='INT',
        default=NUM_TRAIN,
        type=int,
        help='Number of total images to select for training')
    add('--n_test',
        metavar='INT',
        default=NUM_TEST,
        type=int,
        help='Number of total images to select for validation')
    add('-wd', '--weight_decay',
        metavar='FLOAT',
        default=WEIGHT_DEACY,
        type=float,
        help='Weight decay for sgd_th')
    add('-reg', '--network_regularization',
        metavar='FLOAT',
        default=REGULARIZATION,
        type=float,
        help='L2 regularization term for the network')
    add('-lr', '--learning_rate',
        metavar='FLOAT',
        default=LEARNING_RATE,
        type=float,
        help='Learning rate to use with sgd_th')
    add('-mom', '--momentum',
        metavar='FLOAT',
        default=MOMENTUM,
        type=float,
        help='Nesterov momentum use with sgd_th')
    add('--n_epochs', '-nep',
        metavar='INT',
        default=NUM_EPOCHS,
        type=int,
        help='Number of epochs for training')
    add('--batch_size', '-bs',
        metavar='INT',
        default=BATCH_SIZE,
        type=int,
        help='Number of images for each iteration')
    add('--checkpoint_every', '-cp',
        metavar='INT',
        default=CHECKPOINT_EVERY,
        type=int,
        help='Number of epochs between each checkpoint')
    parser.parse_args(namespace=args)
    assert not (args.network_regularization and args.weight_decay)


def data_augm(batch):
    p = 2
    h, w = XH, XW

    # batch = random_tint(batch)
    # batch = random_contrast(batch)
    batch = random_flips(batch)
    # batch = random_rotate(batch, 10)
    batch = random_crops(batch, (h, w), pad=p)
    return batch


def custom_update_decay(epoch):
    if epoch in (80, 120):
        return 0.1
    return 1


def print_infos(solver):
    print 'Model: \n%s' % solver.model

    print 'Solver: \n%s' % solver

    print 'Data Augmentation Function: \n'
    print ''.join(['\t' + i for i in inspect.getsourcelines(data_augm)[0]])
    print 'Custom Weight Decay Update Rule: \n'
    print ''.join(['\t' + i for i in inspect.getsourcelines(custom_update_decay)[0]])


def main():
    parse_args()

    data = get_CIFAR10_data(args.dataset_path,
                            num_training=args.n_train, num_validation=0, num_test=args.n_test)

    data = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_test'],
        'y_val': data['y_test'],
    }

    exp_path = args.experiment_path
    nf = args.n_starting_filters
    reg = args.network_regularization

    model = ResNet(n_size=args.n_size,
                   num_starting_filters=nf,
                   reg=reg)

    wd = args.weight_decay
    lr = args.learning_rate
    mom = args.momentum

    optim_config = {'learning_rate': lr, 'nesterov': True,
                    'momentum': mom, 'weight_decay': wd}

    epochs = args.n_epochs
    bs = args.batch_size
    num_p = args.n_processes
    cp = args.checkpoint_every
    criterion = ClassNLLCriterion()
    solver = Solver(model, data, args.load_checkpoint,
                    criterion=criterion,
                    num_epochs=epochs, batch_size=bs,  # 20
                    update_rule='sgd_th',
                    optim_config=optim_config,
                    custom_update_ld=custom_update_decay,
                    batch_augment_func=data_augm,
                    checkpoint_every=0,
                    num_processes=1)

    print_infos(solver)
    solver.train()

    solver.export_model(exp_path)
    solver.export_histories(exp_path)

    print 'finish'


if __name__ == '__main__':
    main()
