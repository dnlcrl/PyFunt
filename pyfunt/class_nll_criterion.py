from criterion import Criterion
import numpy as np


class ClassNLLCriterion(Criterion):

    """docstring for ClassNLLCriterion"""

    def __init__(self, weights=None, size_average=None):
        super(ClassNLLCriterion, self).__init__()
        if size_average:
            self.size_average = size_average
        else:
            self.size_average = True

        if weights:
            # assert(weights:dim() == 1, "weights input should be 1-D Tensor")
            self.weights = weights
        self.output_tensor = np.zeros(1)
        self.total_weight_tensor = np.ones(1)
        self.target = np.zeros(1)  # , dtype=np.long)

    def __len__(self):
        if self.weights:
            return len(self.weights)
        else:
            return 0

    def update_output(self, x, target):

        # probs=np.exp(scores - np.max(scores, axis=1, keepdims=True))
        # probs /= np.sum(probs, axis=1, keepdims=True)
        # return probs
        # # N = x.shape[0]
        # # loss = -np.mean(self.logp[np.arange(N), target])
        # # self.output = -x
        # # return loss

        # N = x.shape[0]
        # xdev = x - x.max(1, keepdims=True)
        # self.logp = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        # loss = -np.mean(self.logp[np.arange(N), target])
        # self.output = loss
        # import pdb; pdb.set_trace()
        # return self.output

        self.output = - np.mean(x[np.arange(x.shape[0]), target])
        return self.output

    def update_grad_input(self, x, target):
        N = x.shape[0]
        dx = np.exp(x)
        dx[np.arange(N), target] -= 1
        dx /= N
        self.grad_input = dx
        #import pdb; pdb.set_trace()
        return self.grad_input
