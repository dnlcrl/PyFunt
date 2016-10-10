from module import Module


class View(Module):

    def __init__(self, shape):
        super(View, self).__init__()
        if type(shape) is not tuple:
            shape = (shape,)
        self.shape = shape

    def update_output(self, x):
        self.output = x.view().reshape((x.shape[0],) + self.shape)
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = grad_output.view().reshape(x.shape)
        return self.grad_input

    def reset(self):
        pass


# class View(Module):

#     def __init__(self, args):
#         super(View, self).__init__()
#         self.reset_size(args)
#         self.num_input.ndim = None

#     def reset_size(self, args):
#         if len(args) == 1 and type(args[0]) == 'float64':
#             self.size = args[0]
#         else:
#             self.size = None
#         self.num_elements = 1
#         inferdim = False
#         for i in xrange(self.size):
#             szi = self.size[i]
#             if szi >= 0:
#                 self.num_elements *= self.size[i]
#             else:
#                 if szi != -1:
#                     raise Exception('size should be positive or -1')
#                 if inferdim:
#                     raise Exception('only one dimension can be at -1')
#                 inferdim = True

#     def update_output(self, x):
#         self.output = self.output or np.zeros_like(x)
#         batch_size = None
#         if batch_size:
#             self.output = x.view(batch_size, *self.size)
#         else:
#             self.output = x.view(self.size)
#         return self.output

#     def update_grad_input(self, x, grad_output):
#         self.grad_input = self.grad_input or np.zeros_like(grad_output)
#         self.grad_input = grad_output.view(x.size)
#         return self.grad_input

#     def __str__(self):
#         return '%s(%s)' % (type(self), self.size)

#     def reset(self):
#         pass
