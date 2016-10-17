from criterion import Criterion
import numpy as np

EPS = 1e-12


class BCECriterion(Criterion):

    """docstring for ClassNLLCriterion"""

    def __init__(self, weights=None, size_average=None):
        super(BCECriterion, self).__init__()
        if size_average:
            self.size_average = size_average
        else:
            self.size_average = True

        if weights:
            # assert(weights:dim() == 1, "weights input should be 1-D Tensor")
            self.weights = weights

    def __len__(self):
        if self.weights:
            return len(self.weights)
        else:
            return 0

    def update_output(self, x, target):
        self.output = - \
            np.mean(np.log(x + EPS) + np.log(1. - x + EPS) * (1. - target))
        # x[np.arange(x.shape[0]), target])

        return self.output

    def update_grad_input(self, x, target):
        N = x.shape[0]
        target.shape = x.shape
        norm = 1./N if self.size_average else 1.
        dx = - norm * (target - x) / ((1. - x + EPS) * (x + EPS))
        self.grad_input = dx
        return self.grad_input
