import mxnet as mx

def conv3x3(data, num_filter, stride=1, padding=1, dilation=1):
    return mx.sym.Convolution(data=data, kernel=(3,3), num_filter=num_filter, stride=(stride, stride), dilate=(dilation, dilation), pad=(padding, padding))


def basic_block(data, num_filter, stride=1, downsample=None, dilation=(1, 1), residual=True):
    out = data
    out = conv3x3(out, num_filter, stride, padding=dilation[0], dilation=dilation[0])
    out = mx.sym.BatchNorm(data=out, fix_gamma=True)
    out = mx.sym.Activation(data=out, act_type='relu')

    out = conv3x3(out, num_filter, padding=dilation[1], dilation=dilation[1])
    out = mx.sym.BatchNorm(data=out, fix_gamma=True)

    if residual:
        _residual = data
        if downsample is not None:
            _residual = downsample(data=_residual)
        out = _residual + out

    out = mx.sym.Activation(data=out, act_type='relu')

    return out


def drn_unit(data, block, in_channel, num_filter, layers, stride=1, dilation=1, new_level=True, residual=True):
    assert dilation == 1 or dilation % 2 == 0
    if block == basic_block:
        block_expansion = 1
    downsample = None
    if stride != 1 or in_channel != num_filter * block_expansion:
        downsample = lambda data: \
            mx.sym.BatchNorm(data=mx.sym.Convolution(data=data, num_filter=num_filter*block_expansion, kernel=(1,1), stride=(stride, stride), no_bias=True), fix_gamma=True)

    out = block(data, num_filter, stride, downsample, dilation=(1,1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation), residual=residual)

    return out

    for i in range(1, layers):
        out = block(out, num_filter, residual=residual, dilation=(dilation, dilation))

    return out


def drn(arch, block, layers, num_classes=1000, channels=(16,32,64,128,256,512,512,512)):
    data = mx.sym.Variable(name='data')
    if arch=='C':
        out = data
        out = mx.sym.Convolution(data=out, num_filter=channels[0], kernel=(7,7), stride=(1,1), pad=(3,3), no_bias=True)
        out = mx.sym.BatchNorm(data=out, fix_gamma=True)
        out = mx.sym.Activation(data=out, act_type='relu')

        out = drn_unit(out, basic_block, channels[0], channels[0], layers[0], stride=1)
        out = drn_unit(out, basic_block, channels[0], channels[1], layers[1], stride=2)
        num_channel = channels[1]
    else:
        raise NotImplementedError()

    out = drn_unit(out, block, num_channel, channels[2], layers[2], stride=2)
    out = drn_unit(out, block, channels[2], channels[3], layers[3], stride=2)
    out = drn_unit(out, block, channels[3], channels[4], layers[4], dilation=2, new_level=False)

    num_channel = channels[4]
    if layers[5] > 0:
        out = drn_unit(out, block, num_channel, channels[5], layers[5], dilation=4, new_level=False)
        num_channel = channels[5]

    if arch == 'C':
        if layers[6] > 0:
            out = drn_unit(out, block, num_channel, channels[6], layers[6], dilation=2, new_level=False, residual=False)
            num_channel = channels[6]
        if layers[7] > 0:
            out = drn_unit(out, block, num_channel, channels[7], layers[7], dilation=1, new_level=False, residual=False)
            num_channel = channels[7]

    return out


def get_symbol(arch='C', num_layers='26'):
    if arch == 'C':
        if num_layers == 26:
            fblock = basic_block
            layers = [1, 1, 2, 2, 2, 2, 1, 1]
    else:
        raise NotImplementedError()

    net = drn(arch, fblock, layers)

    return net
