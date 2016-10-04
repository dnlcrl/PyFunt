from container import Container


class Parallel(Container):
    """docstring for Parallel"""
    def __init__(self):
        super(Parallel, self).__init__()

    def len(self):
        return len(self.modules)

    def add(self,  module):
        pass

    def insert(self, modules, module):
        pass

    def remove(self, index):
        pass

    def update_output(self, x):
        pass

    def update_grad_input(self, grad_output):
        pass

    def acc_grad_parameters(self, grad_output, scale):
        pass

    def backward(self, grad_output, scale):
        pass

    def __str__(self):
        pass
