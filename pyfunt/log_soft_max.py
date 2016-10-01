from module import Module
import numpy as np


class LogSoftMax(Module):
    """docstring for LogSoftMax"""
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def update_output(self, x):
        # max_input = x.max(1, keepdims=True)
        # log_sum = np.sum(np.exp(x - max_input))
        # log_sum = max_input + np.log(log_sum)
        # output = x - log_sum
        # import pdb; pdb.set_trace()
        # self.output = output
        # if not hasattr(self, 'x'):
        #     self.x = self.output

        # else:
        #     if np.all(self.x == self.output):
        #         import pdb; pdb.set_trace()
        #return self.output
        xdev = x - x.max(1, keepdims=True)
        logp = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        self.output = logp
        self.logp = logp
        return logp

    def upgrade_grad_input(self, x, grad_output):
        # xdev = x - x.max(1, keepdims=True)
        # logp = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))

        # dx = np.exp(logp)

        # s = np.sum(grad_output)
        # grad_input = grad_output - np.exp(self.output)*s
        # self.grad_input = grad_input
        # return self.grad_input

        # dx = np.exp(self.logp)
        # dx[np.arange(N), y] -= 1
        # dx /= N
        # self.grad_input = dx
        # self.output = loss
        # return self.grad_input
        self.grad_input = self.grad_output / x.shape[0]
        return self.grad_input
