import torchfile
import pyfunt

o = torchfile.load('/Users/mbp/Downloads/Chrome/vgg16.t7')
for node in o.forwardnodes: print(repr(node.data.module))

for tmodule in o.modules:
    if type(tmodule) is torchfile.TorchObject:

