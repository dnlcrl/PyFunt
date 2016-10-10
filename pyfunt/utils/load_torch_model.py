import torchfile
import pyfunt
import pdb
import re

please_contribute = 'If you want you can fix it and make a pull request ;)'


'''
<Layer>_init (module) takes a dict for the torch layer and returns a tuple
containing the values for the pyfunt layer initialization funciton.
Once you wrote the function, add the reation in the load_parser_init dict.
The same mechanism goes for the layer values using load_parser_vals dictt
(gard input, output, weight, bias already get added).
'''


def conv_init(m):
    return m['nInputPlane'], m['nOutputPlane'], m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def dropout_init(m):
    return m['p'], not m['v2']


def linear_init(m):
    return m['weight'].shape[1], m['weight'].shape[0], len(m['bias']) != 0


def mul_constant_init(m):
    return (m['constant_scalar'],)


def relu_init(m):
    return (m['inplace'],)


def spatial_max_pooling_init(m):
    return m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def spatial_average_pooling_init(m):
    return m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def spatial_full_convolution_init(m):
    return m['nInputPlane'], m['nOutputPlane'], m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH'], m['adjW'], m['adjH']


def spatial_padding_init(m):
    return m['pad_l'], m['pad_r'], m['pad_t'], m['pad_b']


def view_init(m):
    return (m['size'],)


load_parser_init = {
    'Dropout': dropout_init,
    'Linear': linear_init,
    'MulConstant': mul_constant_init,
    'ReLU': relu_init,
    'SpatialConvolution': conv_init,
    'SpatialMaxPooling': spatial_max_pooling_init,
    'SpatialAvergaePooling': spatial_average_pooling_init,
    'SpatialFullConvolution': spatial_full_convolution_init,
    'SpatialReflectionPadding': spatial_padding_init,
    'SpatialReplicationPadding': spatial_padding_init,
    'View': view_init
}


# def add_possible_values(module, tmodule):
#     print(len(dir(tmodule)))
#     for k in dir(tmodule):
#         if any(x.isupper() for x in k):
#             ourk = re.sub('([A-Z]+)', r'_\1', k).lower()
#             add_value(module, tmodule, ourk, k)
#         else:
#             add_value(module, tmodule, k)


def dropout_vals(module, tmodule):
    add_value(module, tmodule, 'noise')


def spatial_batch_normalization_vals(module, tmodule):
    add_value(module, tmodule, 'running_mean')
    add_value(module, tmodule, 'running_var')


load_parser_vals = {
    'Droput': dropout_vals,
    'SpatialBatchNormalization': spatial_batch_normalization_vals
}


def load_t7model(path=None, obj=None, model=None, custom_layers=None):
    if not (path is None or obj is None):
        raise Exception('you must pass a path or a TorchObject')
    if path:
        o = torchfile.load(path)
    else:
        o = obj

    # import pdb; pdb.set_trace()
    if type(o) is torchfile.TorchObject:
        class_name = o._typename.split('.')[-1]
        tmodule = o._obj

        if not hasattr(pyfunt, class_name):
            print('class %s not found' % class_name)
            print(please_contribute)
            raise NotImplementedError

        Module = getattr(pyfunt, class_name)
        if not is_container(Module):
            raise Exception('model is a torchobj but not a container')
        model = Module()
        add_inout(model, tmodule)

        m = load_t7model(obj=tmodule, model=model, custom_layers=custom_layers)
        if not model:
            model = m
    else:

        for i, tmodule in enumerate(o.modules):
            if type(tmodule) is torchfile.TorchObject:
                class_name = tmodule._typename.split('.')[-1]
                tmodule_o = tmodule._obj

                if hasattr(pyfunt, class_name):
                    Module = getattr(pyfunt, class_name)
                elif custom_layers and hasattr(custom_layers, class_name):
                    Module = getattr(custom_layers, class_name)
                else:
                    print('class %s not found' % class_name)
                    print(please_contribute)
                    raise NotImplementedError

                if i == 0 and model is None:
                    if not is_container(Module):
                        model = pyfunt.Sequential()
                #     else:
                #         model = Module()
                #         model = load_t7model(obj=tmodule, model=model)
                # else:
                if is_container(Module):
                    model.add(
                        load_t7model(obj=tmodule, model=model, custom_layers=custom_layers))
                else:
                    if class_name in load_parser_init:
                        args = load_parser_init[class_name](tmodule_o)
                        module = Module(*args)
                    else:
                        try:
                            module = Module()
                        except:
                            print('parser for %s not found' % class_name)
                            print('%s cannot be initialized with no args' %
                                  class_name)
                            print(please_contribute)
                            raise NotImplementedError

                    #add_possible_values(module, tmodule)
                    add_inout(module, tmodule_o)
                    add_w(module, tmodule_o)
                    if class_name in load_parser_vals:
                        load_parser_vals[class_name](module, tmodule_o)

                    model.add(module)
            else:
                print('oops!')
                print(please_contribute)
                pdb.set_trace()
                raise NotImplementedError
    return model


def is_container(tm):
    return pyfunt.container.Container in tm.__bases__


def add_value(module, tmodule, pname, tpname=None):
    tpname = tpname or pname
    if hasattr(module, pname):
        if tpname in tmodule:
            setattr(module, pname, tmodule[tpname])


def add_inout(module, tmodule):
    add_value(module, tmodule, 'output')
    add_value(module, tmodule, 'grad_input', 'gradInput')


def add_w(module, tmodule):
    add_value(module, tmodule, 'weight')
    add_value(module, tmodule, 'bias')
    add_value(module, tmodule, 'grad_weight', 'gradWeight')
    add_value(module, tmodule, 'grad_bias', 'gradBias')


def load_t7checkpoint(path, models_keys=['model'], custom_layers=None):
    # model_keys iterable that contains for example the word 'model'
    # the model to load in pyfunt
    cp = torchfile.load(path)
    for model in models_keys:
        cp[model] = load_t7model(obj=cp[model], custom_layers=custom_layers)
    return cp
