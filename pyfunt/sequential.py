from container import Container


class Sequential(Container):
    """docstring for Sequential"""
    def __init__(self):
        super(Sequential, self).__init__()

    def len(self):
        return len(self.modules)

    def add(self,  module):
        if len(self.modules) == 0:
            self.grad_input = module.grad_input
        self.modules.insert(module)
        self.output = module.output
        return self

    def insert(self, module, index=None):
        index = index or len(self.modules) + 1
        if index > len(self.modules) + 1 or index < 1:
            raise('index should be contiguous to existing modules')
        self.modules.insert(module, index)
        self.output = self.modules[len(self.modules)].output
        self.grad_input = self.modules[0].grad_input ## 1??

    def remove(self, index):
        if index > len(self.modules) or index < 1:
            raise('index out of range')
        self.modules.remove(index)
        if len(self.modules) > 0:
            self.output = self.modules[len(self.modules)].output
            self.grad_input = self.modules[0].grad_input
        else:
            self.output = mp.ndarray()
            self.grad_input = mp.ndarray()

    def update_output(self, x):
        current_output = x
        for i in xrange(len(self.modules)):
            current_output = self.rethrow_errors(self.modules[i], i, 'update_output', current_output)
        self.output = current_output
        return self.output

    def update_grad_input(self, grad_output):
        current_grad_output = grad_output
        current_module = self.modules[len(self.modules)]
        for i in range(len())# reverse cicle


    def acc_grad_parameters(self, grad_output, scale):
        pass

    def backward(self, grad_output, scale):
        pass

    def __str__(self):
        pass
