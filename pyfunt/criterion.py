import numpy as np
import abc


class Criterion(object):
    __metaclass__ = abc.ABCMeta
    """docstring for Criterion"""
    def __init__(self):
        super(Criterion, self).__init__()
        self.output = 0

    @abc.abstractmethod
    def update_output(self, x, target):
        pass

    def forward(self, x, target):
        return self.update_output(x, target)

    def backward(self, x, target):
        return self.update_grad_input(x, target)

    @abc.abstractmethod
    def update_grad_input(self, x, target):
        pass

    def clone(self):
        pass

    def __call__(self, x, target):
        self.output = self.forward(x, target)
        self.grad_input = self.backward(x, target)
        return self.output, self.grad_input
