from container import Container


class ConcatTable(Container):

    """docstring for ConcatTable"""

    def __init__(self):
        super(ConcatTable, self).__init__()
        self.modules = {}
        self.output = {}

    def update_output(input):
        for i in xrange(len(self.modules)):
            self.output[i] = None  # concat

    def upgrade_grad_input(self, x, grad_output):
        #unpad
        return self.grad_input
