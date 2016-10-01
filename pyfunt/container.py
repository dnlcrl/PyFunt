from module import Module
from types import DictType
import sys
import traceback
import numpy as np

class Container(Module):
    """docstring for Container"""
    def __init__(self):
        super(Container, self).__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)
        return self

    def get(self, index):
        return self.modules[index]

    def size(self):
        return len(self.modules)

    def rethrow_errors(self, module, module_index, func_name, *args):
        def handle_error(err):
            # TODO
            return err
        func = getattr(module, func_name)
        try:
            result = func(*args)
        except Exception as e:
            print 'In %d module (%s) of %s:' % (module_index, type(module).__name__, type(self).__name__)
            traceback.print_exc()
            raise e

        return result

    def apply_to_modules(self, func):
        for module in self.modules:
            func(module)

    def zero_grad_parameters(self):
        self.apply_to_modules(lambda module: module.zero_grad_parameters())

    def update_parameters(self, lr):
        self.apply_to_modules(lambda module: module.update_parameters(lr))

    def training(self):
        self.apply_to_modules(lambda module: module.training())
        super(Container, self).training(self)

    def evaluate(self):
        self.apply_to_modules(lambda module: module.evaluate())
        super(Container, self).evaluate(self)

    def share(self, mlp, args):
        pass

    def reset(self, stdv):
        self.apply_to_modules(lambda module: module.reset(stdv))

    def parameters(self):
        def tinsert(to, _from):
            if isinstance(_from, list):
                for i in xrange(len(_from)):
                    tinsert(to, _from[i])
            else:
                to.append(_from)

        w = []
        gw = []
        for i in xrange(len(self.modules)):

            res = self.modules[i].parameters()
            if res:
                mw, mgw = res
                tinsert(w, mw)
                tinsert(gw, mgw)
        return w, gw

    def clear_state(self):
        pass
