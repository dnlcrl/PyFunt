from module import Module
from types import DictType


class Container(Module):
    """docstring for Container"""
    def __init__(self):
        super(Container, self).__init__()

    def add(self, module):
        self.modules.update(module)
        return self

    def get(self, index):
        return self.modules[index]

    def size(self):
        return len(self.modules)

    def apply_to_modules(self, func):
        for module in self.modules.values:
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
            if type(_from) == DictType:
                for i in xrange(len(_from)):
                    tinsert(to, _from[i])
            else:
                to.update(_from)

        w = {}
        gw = {}
        for i in xrange(len(self.modules)):
            mw, mgw = self.modules[i].parameters()
            if mw:
                tinsert(w, mw)
                tinsert(gw, mgw)
        return w, gw

    def clear_state(self):
        pass
