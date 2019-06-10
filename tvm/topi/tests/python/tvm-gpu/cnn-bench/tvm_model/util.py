import tvm
import nnvm
import nnvm.testing
import numpy as np
from nnvm import sym

def int8_wrapper(f, data, **kwargs):
    data = sym.cast(data=data, dtype='float32')
    data = f(data=data, **kwargs)
    data = sym.cast(data=data, dtype='int8')
    return data

def create_workload(net, batch_size, image_shape=(3, 224, 224),
                    dtype="float32", initializer=None, seed=0):
    """Helper function to create benchmark workload for input network

    Parameters
    ----------
    net : nnvm.Symbol
        The selected network symbol to use

    batch_size : int
        The batch size used in the model

    image_shape : tuple, optional
        The input image shape

    dtype : dict, optional
        The data type

    initializer : Initializer
        The initializer used

    seed : int
        The seed used in initialization.

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.

    dtype : dict of str to str
        dtype of each parameter
    """
    if image_shape is None:
        image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    params = {}
    g = nnvm.graph.create(net)
    input_shapes, _ = nnvm.compiler.graph_util.infer_shape(g, data=data_shape)
    shape_dict = dict(zip(g.index.input_names, input_shapes))
    """
    for key in shape_dict.keys():
        if("weight" in key):
            temp=shape_dict[key][1]
            shape_dict[key][1]=shape_dict[key][2]
            shape_dict[key][2]=shape_dict[key][3]
            shape_dict[key][3]=temp
    """
    np.random.seed(seed)
    initializer = initializer if initializer else nnvm.testing.init.Xavier()
    for k, v in shape_dict.items():
        if k == "data":
            continue
        if isinstance(dtype, dict):
            _dtype = dtype[k]
        else:
            _dtype = dtype
        init_value = np.random.uniform(0.,1./512.,v).astype(_dtype)
        #initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
        
    return net, params

def create_int8_network(fsymbol, batch_size, num_classes=1000, image_shape=(3, 224, 224), **kwargs):
    net = fsymbol(num_classes=num_classes, **kwargs)

    g = nnvm.graph.create(net)
    input_names = g.index.input_names
    dtype = {name: 'int8' if ('bn' not in name and 'batch_norm' not in name) else 'float32' for name in input_names}
    net, params = create_workload(net, batch_size, dtype=dtype, image_shape=image_shape)
    return net, params, dtype

def create_fp16_network(fsymbol, batch_size, num_classes=1000, image_shape=(256,256,64), **kwargs):
    net = fsymbol(num_classes=num_classes, **kwargs)
    g = nnvm.graph.create(net)
    input_names = g.index.input_names
    dtype = {name: 'float16' for name in input_names}
    net, params = create_workload(net, batch_size, dtype=dtype, image_shape=image_shape)

    return net, params, dtype
