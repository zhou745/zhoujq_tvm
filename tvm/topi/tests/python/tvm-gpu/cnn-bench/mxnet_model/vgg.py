import mxnet as mx

def get_feature(internel_layer, layers, filters, batch_norm = False, **kwargs):
    for i, num in enumerate(layers):
        for j in range(num):
            internel_layer = mx.sym.Convolution(data = internel_layer, kernel=(3, 3), pad=(1, 1), num_filter=filters[i], name="conv%s_%s" %(i + 1, j + 1))
            if batch_norm:
                internel_layer = mx.symbol.BatchNorm(data=internel_layer, name="bn%s_%s" %(i + 1, j + 1))
            internel_layer = mx.sym.Activation(data=internel_layer, act_type="relu", name="relu%s_%s" %(i + 1, j + 1))
        internel_layer = mx.sym.Pooling(data=internel_layer, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool%s" %(i + 1))
    return internel_layer

def get_symbol(num_layers=11, batch_norm=False, dtype='float32', **kwargs):
    """
    Parameters
    ----------
    num_classes : int, default 1000
        Number of classification classes.
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    batch_norm : bool, default False
        Use batch normalization.
    dtype: str, float32 or float16
        Data precision.
    """
    vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
                13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
                16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
                19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}
    if num_layers not in vgg_spec:
        raise ValueError("Invalide num_layers {}. Possible choices are 11,13,16,19.".format(num_layers))
    layers, filters = vgg_spec[num_layers]
    data = mx.sym.Variable(name="data")
    if dtype != 'float32':
        data = mx.sym.Cast(data=data, dtype=dtype)
    feature = get_feature(data, layers, filters, batch_norm)
    return feature

def get_network(batch_size, dtype):
    image_shape = (3, 224, 224)
    net = get_symbol(num_layers=16, batch_norm=False, dtype=dtype)

    return (net, [('data', (batch_size,) + image_shape)])

