from module import Module
import numpy as np


class Dropout(Module):

    def __init__(self, p=0.5, v1=False, stochastic_inference=False):
        super(Dropout, self).__init__()
        self.p = p
        self.train = True
        self.stochastic_inference = stochastic_inference
        # version 2 scales output during training instead of evaluation
        self.v2 = not v1
        if self.p >= 1 or self.p < 0:
            raise('<Dropout> illegal percentage, must be 0 <= p < 1')
        self.noise = None

    def update_output(self, x):
        self.output = x.copy()
        if self.p > 0:
            if self.train or self.stochastic_inference:
                self.noise = np.zeros_like(x)
                self.noise = np.random.binomial(1, p=1-self.p, size=x.shape)  # bernoulli
                if self.v2:
                    self.noise /= 1-self.p
                self.output *= self.noise
            elif not self.v2:
                self.output *= 1-self.p
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = grad_output.copy()
        if self.train:
            if self.p > 0:
                self.grad_input *= self.noise
        else:
            if not self.v2 and self.p > 0:
                self.grad_input *= 1-self.p
        return self.grad_input

    def __str__(self):
        return '%s(%f)' % (type(self), self.p)

    def reset(self):
        pass
