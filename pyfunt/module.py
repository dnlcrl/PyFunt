#!/usr/bin/env python
# coding: utf-8

import abc
import numpy as np
from copy import deepcopy
from types import DictType


class Module(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.grad_input = None  # np.ndarray()
        self.output = None  # np.ndarray()
        self._type = np.float

    def parameters(self):
        if hasattr(self, 'weight'):
            if self.weight is not None and self.bias is not None:
                return [self.weight, self.bias], [self.grad_weight, self.grad_bias]
            if self.weight is not None:
                return [self.weight], [self.grad_weight]
            if self.bias is not None:
                return [self.bias], [self.grad_bias]

    @abc.abstractmethod
    def update_output(self, _input=None):
        # return self.output
        raise NotImplementedError()

    def forward(self, x):
        return self.update_output(x)

    def backward(self, _input, grad_output, scale=1):
        self.grad_input = self.update_grad_input(_input, grad_output)
        self.acc_grad_parameters(_input, grad_output, scale)
        return self.grad_input

    def backward_update(self, _input, grad_output, lr):
        grad_weight = self.grad_weight
        grad_bias = self.grad_bias
        self.grad_weight = self.weight
        self.grad_bias = self.bias
        self.acc_grad_parameters(_input, grad_output, -lr)
        self.grad_weight = grad_weight
        self.grad_bias = grad_bias

    @abc.abstractmethod
    def update_grad_input(self, _input, grad_output):
        # return self.grad_input
        raise NotImplementedError()

    def acc_grad_parameters(self, _input, grad_output, scale):
        pass

    def acc_update_grad_parameters(self, _input, grad_output, lr):
        grad_weight = self.grad_weight
        grad_bias = self.grad_bias
        self.grad_weight = self.weight
        self.grad_bias = self.bias
        self.acc_grad_parameters(_input, grad_output, -lr)
        self.grad_weight = grad_weight
        self.grad_bias = grad_bias

    def shared_acc_update_grad_parameters(self, _input, grad_output, lr):
        if self.parameters():
            self.zero_grad_parameters()
            self.acc_grad_parameters(_input, grad_output, 1)
            self.update_parameters(lr)

    def zero_grad_parameters(self):
        _, grad_params = self.parameters()
        if grad_params:
            for g in grad_params:
                g.zero()

    def update_parameters(self, lr):
        res = self.parameters()
        if res:
            params, grad_params = res
            for i, p in enumerate(params):
                p -= lr*grad_params[i]

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False

    def share(self, mlp, p_names):
        for i, v in enumerate(p_names):
            if self[v] is not None:
                self[v].set(mlp[v])
                self.acc_update_grad_parameters = self.shared_acc_update_grad_parameters
                mlp.acc_update_grad_parameters = self.acc_update_grad_parameters
        return self

    def clone(self, p_names=None):
        clone = deepcopy(self)
        if p_names:
            clone.share(self, p_names)
        return clone

    # def type(self, type=None, cache=None):
    #     if type is not None:
    #         return self._type
    #     cache = cache or {}
    #     # find all tensors and convert them
    #     for key, param in pairs(self):
    #         self[key] = utils.recursive_type(param, type, cache)

    #     self._type = type
    #     return self


# function Module:float(...)
#    return self:type('torch.FloatTensor',...)
# end

# function Module:double(...)
#    return self:type('torch.DoubleTensor',...)
# end

# function Module:cuda(...)
#    return self:type('torch.CudaTensor',...)
# end

    def reset(self):
        raise NotImplementedError()

    def write(self, file):
        np.save(file, self)

    def read(self, file):
        obj = np.load(file)[0]
        for k, v in enumerate(obj):
            self[k] = v


# -- This function is not easy to understand. It works as follows:
# --
# -- - gather all parameter tensors for this module (and children);
# --   count all parameter values (floats)
# -- - create one ginormous memory area (Storage object) with room for all
# --   parameters
# -- - remap each parameter tensor to point to an area within the ginormous
# --   Storage, and copy it there
# --
# -- It has the effect of making all parameters point to the same memory area,
# -- which is then returned.
# --
# -- The purpose is to allow operations over all parameters (such as momentum
# -- updates and serialization), but it assumes that all parameters are of
# -- the same type (and, in the case of CUDA, on the same device), which
# -- is not always true. Use for_each() to iterate over this module and
# -- children instead.
# --
# -- Module._flattenTensorBuffer can be used by other packages (e.g. cunn)
# -- to specify the type of temporary buffers. For example, the temporary
# -- buffers for CudaTensor could be FloatTensor, to avoid GPU memory usage.
# --
# -- TODO: This logically belongs to torch.Tensor, not nn.
# Module._flattenTensorBuffer = {}


# function Module.flatten(parameters)

#    -- returns true if tensor occupies a contiguous region of memory (no holes)
#    local function isCompact(tensor)
#       local sortedStride, perm = torch.sort(
#             torch.LongTensor(tensor:nDimension()):set(tensor:stride()), 1, true)
#       local sortedSize = torch.LongTensor(tensor:nDimension()):set(
#             tensor:size()):index(1, perm)
#       local nRealDim = torch.clamp(sortedStride, 0, 1):sum()
#       sortedStride = sortedStride:narrow(1, 1, nRealDim):clone()
#       sortedSize   = sortedSize:narrow(1, 1, nRealDim):clone()
#       local t = tensor.new():set(tensor:storage(), 1,
#                                  sortedSize:storage(),
#                                  sortedStride:storage())
#       return t:isContiguous()
#    end

#    if not parameters or #parameters == 0 then
#       return torch.Tensor()
#    end
#    local Tensor = parameters[1].new
# local TmpTensor = Module._flattenTensorBuffer[torch.type(parameters[1])]
# or Tensor

#    -- 1. construct the set of all unique storages referenced by parameter tensors
#    local storages = {}
#    local nParameters = 0
#    local parameterMeta = {}
#    for k = 1,#parameters do
#       local param = parameters[k]
#       local storage = parameters[k]:storage()
#       local storageKey = torch.pointer(storage)

#       if not storages[storageKey] then
#          storages[storageKey] = {storage, nParameters}
#          nParameters = nParameters + storage:size()
#       end

#       parameterMeta[k] = {storageOffset = param:storageOffset() +
#                                           storages[storageKey][2],
#                           size          = param:size(),
#                           stride        = param:stride()}
#    end

#    -- 2. construct a single tensor that will hold all the parameters
#    local flatParameters = TmpTensor(nParameters):zero()

#    -- 3. determine if there are elements in the storage that none of the
#    --    parameter tensors reference ('holes')
#    local tensorsCompact = true
#    for k = 1,#parameters do
#       local meta = parameterMeta[k]
#       local tmp = TmpTensor():set(
#          flatParameters:storage(), meta.storageOffset, meta.size, meta.stride)
#       tmp:fill(1)
#       tensorsCompact = tensorsCompact and isCompact(tmp)
#    end

#    local maskParameters  = flatParameters:byte():clone()
#    local compactOffsets  = flatParameters:long():cumsum(1)
#    local nUsedParameters = compactOffsets[-1]

#    -- 4. copy storages into the flattened parameter tensor
#    for _, storageAndOffset in pairs(storages) do
#       local storage, offset = table.unpack(storageAndOffset)
#       flatParameters[{{offset+1,offset+storage:size()}}]:copy(Tensor():set(storage))
#    end

#    -- 5. allow garbage collection
#    storages = nil
#    for k = 1,#parameters do
#        parameters[k]:set(Tensor())
#    end

#    -- 6. compact the flattened parameters if there were holes
#    if nUsedParameters ~= nParameters then
#       assert(tensorsCompact,
#          "Cannot gather tensors that are not compact")

#       flatParameters = TmpTensor(nUsedParameters):copy(
#             flatParameters:maskedSelect(maskParameters))
#       for k = 1,#parameters do
#         parameterMeta[k].storageOffset =
#               compactOffsets[parameterMeta[k].storageOffset]
#       end
#    end

#    if TmpTensor ~= Tensor then
#       flatParameters = Tensor(flatParameters:nElement()):copy(flatParameters)
#    end

#    -- 7. fix up the parameter tensors to point at the flattened parameters
#    for k = 1,#parameters do
#       parameters[k]:set(flatParameters:storage(),
#           parameterMeta[k].storageOffset,
#           parameterMeta[k].size,
#           parameterMeta[k].stride)
#    end

#    return flatParameters
# end

    def get_parameters(self):
        parameters, grad_parameters = self.parameters()
        #p, g = Module.flatten(parameters), Module.flatten(grad_parameters)
        #if not p.n_element() == g.n_element():
        #    raise Exception('check that you are sharing parameters and gradParameters')
        return parameters, grad_parameters

    def __call__(self, _input=None, grad_output=None):
        self.forward(_input)
        if self.grad_output:
            self.backward(_input, grad_output)
            return self.output, self.grad_input
        else:
            return self.output

    # Run a callback (called with the module as an argument) in preorder over this
    # module and its children.
    def apply(self, callback):
        callback(self)
        if self.modules:
            for module in self.modules:
                module.apply(callback)

    def find_modules(self, type_c, container):
        container = container or self
        nodes = {}
        containers = {}
        mod_type = type(self)
        if mod_type == type_c:
            nodes[len(nodes)+1] = self
            containers[len(containers)] = container
        # Recurse on nodes with 'modules'
        if self.modules is not None:
            if type(self.modules) is DictType:
                for i in xrange(len(self.modules)):
                    child = self.modules[i]
                    cur_nodes, cur_containers = child.find_modules(
                        type_c, self)

                    # This shouldn't happen
                    if not len(cur_nodes) == len(cur_containers):
                        raise Exception('Internal error: incorrect return length')

                    # add the list items from our child to our list (ie return a
                    # flattened table of the return nodes).
                    for j in xrange(len(cur_nodes)):
                        nodes[len(cur_nodes)+1] = cur_nodes[j]
                        containers[len(containers)+1] = cur_containers[j]

        return nodes, containers

    def list_modules(self):
        def tinsert(to, _from):
            if type(_from) == DictType:
                for i in xrange(len(_from)):
                    tinsert(to, _from[i])
            else:
                to.update(_from)

        modules = self
        if self.modules:
            for i in xrange(len(self.modules)):
                modulas = self.modules[i].list_modules()
                if modulas:
                    tinsert(modules, modulas)
        return modules

    def clear_state(self):
        return  # clear utils clear(self, 'output', 'gradInput')

    # similar to apply, recursively goes over network and calls
    # a callback function which returns a new module replacing the old one

    def replace(self, callback):
        callback(self)
        if self.modules:
            for i, m in enumerate(self.modules):
                self.modules[i] = Module.replace(callback)
