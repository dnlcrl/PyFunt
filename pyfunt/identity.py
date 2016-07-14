from module import Module


class Identity(Module):
    """docstring for Identity"""
    def __init__(self):
        super(Identity, self).__init__()

    def update_output(self, x):
        self.output = x
        return self.output

    def update_grad_input(self, grad_output):
        self.grad_input = grad_output
        return self.grad_input

    def clear_state(self):
        pass
