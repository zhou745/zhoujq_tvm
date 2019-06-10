from nnvm import sym
from .util import int8_wrapper

def conv3x3(data, num_filter, stride=1, padding=1, dilation=1):
    return sym.conv2d(data=data, kernel_size=(3,3), channels=num_filter, strides=(stride, stride), dilation=(dilation, dilation), padding=(padding, padding))


def basic_block(data, num_filter, stride=1, downsample=None, dilation=(1, 1), residual=True):
    out = data
    out = conv3x3(out, num_filter, stride, padding=dilation[0], dilation=dilation[0])
    out = int8_wrapper(sym.batch_norm, data=out)
    out = sym.relu(data=out)

    out = conv3x3(out, num_filter, padding=dilation[1], dilation=dilation[1])
    out = int8_wrapper(sym.batch_norm, data=out)

    if residual:
        _residual = data
        if downsample is not None:
            _residual = downsample(data=_residual)
        out = _residual + out

    out = sym.relu(data=out)

    return out


def drn_unit(data, block, in_channel, num_filter, layers, stride=1, dilation=1, new_level=True, residual=True):
    assert dilation == 1 or dilation % 2 == 0
    if block == basic_block:
        block_expansion = 1
    downsample = None
    if stride != 1 or in_channel != num_filter * block_expansion:
        downsample = lambda data: \
            int8_wrapper(sym.batch_norm, data=sym.conv2d(data=data, channels=num_filter*block_expansion, kernel_size=(1,1), strides=(stride, stride), use_bias=False))

    out = block(data, num_filter, stride, downsample, dilation=(1,1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation), residual=residual)


    for i in range(1, layers):
        out = block(out, num_filter, residual=residual, dilation=(dilation, dilation))

    return out


def drn(arch, block, layers, num_classes=1000, channels=(16,32,64,128,256,512,512,512)):
    data = sym.Variable(name='data')
    if arch=='C':
        out = data
        out = sym.conv2d(data=out, channels=channels[0], kernel_size=(7,7), strides=(1,1), padding=(3,3), use_bias=False)
        out = int8_wrapper(sym.batch_norm, data=out)
        out = sym.relu(data=out)

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


def get_symbol(arch='C', num_layers='26', num_classes=1000):
    if arch == 'C':
        if num_layers == 26:
            fblock = basic_block
            layers = [1, 1, 2, 2, 2, 2, 1, 1]
    else:
        raise NotImplementedError()

    net = drn(arch, fblock, layers, num_classes=num_classes)

    return net
