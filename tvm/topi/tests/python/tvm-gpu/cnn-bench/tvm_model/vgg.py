# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
from nnvm import sym
from .util import int8_wrapper

def get_feature(internel_layer, layers, filters, batch_norm=False):
    """Get VGG feature body as stacks of convoltions."""
    for i, num in enumerate(layers):
        for j in range(num):
            internel_layer = sym.conv2d(
                data=internel_layer, kernel_size=(3, 3), padding=(1, 1),
                channels=filters[i], name="conv%s_%s"%(i + 1, j + 1),use_bias=False)

            #internel_layer = sym.relu(data=internel_layer, name="relu%s_%s" %(i + 1, j + 1))

        internel_layer = sym.max_pool2d(
            data=internel_layer, pool_size=(2, 2), strides=(2, 2), name="pool%s"%(i + 1),layout="NHWC")
    return internel_layer

def get_classifier(input_data, num_classes):
    """Get VGG classifier layers as fc layers."""
    flatten = sym.flatten(data=input_data, name="flatten")
    fc6 = sym.dense(data=flatten, units=4096, name="fc6")
    relu6 = sym.relu(data=fc6, name="relu6")
    drop6 = sym.dropout(data=relu6, rate=0.5, name="drop6")
    fc7 = sym.dense(data=drop6, units=4096, name="fc7")
    relu7 = sym.relu(data=fc7, name="relu7")
    drop7 = sym.dropout(data=relu7, rate=0.5, name="drop7")
    fc8 = sym.dense(data=drop7, units=num_classes, name="fc8")
    return fc8

def get_symbol(num_classes, num_layers=11, batch_norm=False):
    """
    Parameters
    ----------
    num_classes : int, default 1000
        Number of classification classes.
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    batch_norm : bool, default False
        Use batch normalization.
    """
    vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
                13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
                16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
                19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]),
                1:([1,0,0,0,0],[64,0,0,0,0])}
    if num_layers not in vgg_spec:
        raise ValueError("Invalide num_layers {}. Choices are 11,13,16,19.".format(num_layers))
    layers, filters = vgg_spec[num_layers]
    data = sym.Variable(name="data")
    feature = get_feature(data, layers, filters, batch_norm)

    return feature
