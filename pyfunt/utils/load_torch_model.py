import torchfile
import pyfunt
import pdb

please_contribute = 'If you want you can fix it and make a pull request ;)'


def conv_init(m):
    return m['nInputPlane'], m['nOutputPlane'], m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def dropout_init(m):
    return m['p'], not m['v2']


def linear_init(m):
    return m['weight'].shape[1], m['weight'].shape[0], len(m['bias']) != 0


def relu_init(m):
    return (m['inplace'],)


def spatial_max_pooling_init(m):
    return m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def spatial_average_pooling_init(m):
    return m['kW'], m['kH'], m['dW'], m['dH'], m['padW'], m['padH']


def view_init(m):
    return (m['size'],)


parser_init = {
    'SpatialConvolution': conv_init,
    'Dropout': dropout_init,
    'Linear': linear_init,
    'ReLU': relu_init,
    'SpatialMaxPooling': spatial_max_pooling_init,
    'SpatialAvergaePooling': spatial_average_pooling_init,
    'View': view_init
}


def dropout_vals(module, tmodule):
    add_value(module, tmodule, 'noise')


parser_vals = {
    'Droput': dropout_vals
}


def load_t7model(path=None, obj=None, model=None):
    if not (path is None or obj is None):
        raise Exception('you must pass a path or a TorchObject')
    if path:
        o = torchfile.load(path)
    else:
        o = obj

    if type(o) is torchfile.TorchObject:
        class_name = o._typename.split('.')[-1]
        tmodule = o._obj

        if not hasattr(pyfunt, class_name):
            print('class %s not found' % class_name)
            print(please_contribute)
            raise NotImplementedError

        Module = getattr(pyfunt, class_name)
        if not is_container(Module):
            raise('model is a torchobj but not a container')
        model = Module()
        add_inout(model, tmodule)
        model = load_t7model(obj=tmodule, model=model)

    else:
        for i, tmodule in enumerate(o.modules):
            if type(tmodule) is torchfile.TorchObject:
                class_name = tmodule._typename.split('.')[-1]
                tmodule = tmodule._obj

                if not hasattr(pyfunt, class_name):
                    print('class %s not found' % class_name)
                    print(please_contribute)
                    raise NotImplementedError

                Module = getattr(pyfunt, class_name)
                if i == 0 and model is None:
                    if not is_container(Module):
                        model = pyfunt.Sequential()
                    else:
                        model = Module()
                else:
                    if class_name in parser_init:
                        args = parser_init[class_name](tmodule)
                        module = Module(*args)
                    else:
                        try:
                            module = Module()
                        except:
                            print('parser for %s not found' % class_name)
                            print('%s cannot be initialized with no args' % class_name)
                            print(please_contribute)
                            raise NotImplementedError

                    add_inout(module, tmodule)
                    add_w(module, tmodule)
                    if class_name in parser_vals:
                        parser_vals[class_name](module, tmodule)
                    model.add(module)
            else:
                print('oops!')
                print(please_contribute)
                pdb.set_trace()
                raise NotImplementedError
    return model


def is_container(tm):
    return tm.__bases__ == pyfunt.Container


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


def load_t7checkpoint(path, models_keys=['model']):
    # model_keys iterable that contains for example the word 'model'
    # the model to load in pyfunt
    cp = torchfile.load(path)
    for model in models_keys:
        cp[model] = load_t7model(obj=cp[model])
    return cp



if __name__ == '__main__':
    # TODO remove path :P
    d = load_t7checkpoint('/Users/mbp/Desktop/fast-neural-style-pyfunt/models/instance_norm/candy.t7')
    import pdb; pdb.set_trace()
