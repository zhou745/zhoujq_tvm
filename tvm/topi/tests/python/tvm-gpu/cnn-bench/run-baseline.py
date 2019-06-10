import os
import mxnet as mx
from mxnet_model import *
import logging
import time

gpu = mx.gpu()
num_batches = 1000


def get_network(name, batch_size, num_classes=1000):
    image_shape = (3, 224, 224)
    if 'resnet' in name:
        n_layer = int(name.split('-')[1])
        net = resnet(num_classes=num_classes, num_layers=n_layer, image_shape=image_shape)
    elif 'vgg' in name:
        n_layer = int(name.split('-')[1])
        net = vgg(num_classes=num_classes, num_layers=n_layer, batch_norm='bn' in name)
    elif name == 'inception_v3':
        image_shape = (3, 299, 299)
        net = inception_v3(num_classes=num_classes)
    elif 'drn' in name:
        components = name.split('-')
        arch = components[1]
        n_layers = int(components[2])
        net =  drn(arch, n_layers)

    else:
        raise NotImplementedError()

    data_shape = [('data', (batch_size,) + image_shape)]

    return net, data_shape


def score(model_name, batch_size):
    dtype='float32'
    net, data_shape = get_network(model_name, batch_size)
    mod = mx.mod.Module(symbol=net, context=gpu)
    print(data_shape)
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=gpu) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    t = (time.time() - tic) / num_batches # time in sec
    return t


def export_onnx(model_name, filename):
    net, data_shape = get_network(model_name, batch_size=1)
    mod = mx.mod.Module(symbol=net, context=gpu)
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    arg_param, aux_param = mod.get_params()
    arg_param.update(aux_param)
    mx.contrib.onnx.mx2onnx.export_model.export_model(net, params=arg_param, input_shape=[data_shape[0][1]], onnx_file_path=filename)

if __name__ == '__main__':
    for name in ['drn-C-26']:#'vgg-16', 'resnet-50', 'inception_v3']:
        #t = score(name, 32)

        #print('mxnet {} {} ms'.format(name, t*1000))

        onnx_filename = '{}.onnx'.format(name)
        export_onnx(name, onnx_filename)

        for test_int8 in [0, 1]:
            os.system('./trt {} {}'.format(onnx_filename, test_int8))
