import os
import nnvm
import nnvm.testing
import nnvm.compiler

import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import numpy as np
import logging
import argparse
from tvm_model import *

from mx_vgg_m import module_vgg as mvg
from collections import namedtuple
import mxnet as mx

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='resnet-50')
parser.add_argument('--n_trial', type=int, default='500')
parser.add_argument('--tune', action='store_true')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--log_file', type=str, default=None)
parser.add_argument('--pretuned', type=str, default=None)

args = parser.parse_args()

logging.getLogger('autotvm').setLevel(logging.DEBUG)

target = tvm.target.cuda()

network = args.network
log_file = args.log_file or "%s.log" % network

tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'n_trial': 300,
    'early_stopping': 900,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=3),
        runner=autotvm.LocalRunner(number=4, repeat=1, timeout=3),
    ),
}

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 256, 256,16)
    output_shape = (batch_size, 1000)
    if "resnet" in name:
        print("create res")
        n_layer = int(name.split('-')[1])
        image_shape=(input_shape[1],input_shape[2],input_shape[3])
        net, params, dtype = create_fp16_network(resnet, num_layers=n_layer, batch_size=batch_size,image_shape=image_shape, batch_norm='bn' in name)
    elif "vgg" in name:
        print("create vgg")
        n_layer = int(name.split('-')[1])
        image_shape=(input_shape[1],input_shape[2],input_shape[3])
        net, params, dtype = create_fp16_network(vgg, num_layers=n_layer, batch_size=batch_size,image_shape=image_shape, batch_norm='bn' in name)
    elif name == "inception_v3":
        print("create inc")
        image_shape =  (3, 299, 299)
        input_shape = (batch_size,) + image_shape
        net, params, dtype = create_int8_network(inception_v3, batch_size=batch_size, image_shape=image_shape)
    elif "drn" in name:
        print("create drn")
        components = name.split('-')
        arch = components[1]
        n_layer = int(components[2])
        net, params, dtype = create_int8_network(drn, batch_size=batch_size, arch=arch, num_layers=n_layer)
    else:
        raise NotImplementedError()

    return net, params, input_shape, output_shape, dtype

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=500,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):

    for i in range(len(tasks)):
        print(tasks[i].args)
        data, kernel, padding, stride,temp, layout, dtype = tasks[i].args

        new_args = (data, kernel, padding, stride,temp, layout, dtype)

        block_factor = 4
        CO, CI, KH, KW = kernel[1]
        if CO % block_factor == 0 and CI % block_factor == 0:
            new_task = autotvm.task.create(tasks[i].name, new_args, tasks[i].target, tasks[i].target_host, 'fp16')
            tasks[i] = new_task

    if args.pretuned is not None:
        pretuned_ctx = autotvm.apply_history_best(args.pretuned)
        _tasks = []
        for task in tasks:
            if pretuned_ctx._query_inside(target, task.workload) is None:
                _tasks.append(task)
            else:
                print('Ignoring {}'.format(task))
        tasks = _tasks
    print(tasks)

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):
    net, params, input_shape, out_shape, dtype = get_network(network, batch_size=args.batch)
    print(net)
    tasks = autotvm.task.extract_from_graph(net, target=target,
                                            shape={'data': input_shape}, dtype=dtype,
                                            symbols=(nnvm.sym.conv2d,))

    if args.tune:
        tune_tasks(tasks, **tuning_opt)
    
    with autotvm.apply_history_best(log_file):
	print("Compile...")
        gen_para=False
        if not gen_para:
            with nnvm.compiler.build_config(opt_level=3):     
                graph, lib, params = nnvm.compiler.build(
                    net, target=target, shape={'data': input_shape}, params=params, dtype=dtype)
        
            # export library
            tmp = tempdir()
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))
  
        # load and save parameters
        ctx = tvm.context(str(target), 0)

        params_tvm = {k: tvm.nd.array(v, ctx) for k, v in params.items()}        
        params_mx = {}
        for k,v in params.items():
            if "conv" in k or "sc" in k:
                params_mx.update({k:mx.nd.array(np.float32(np.transpose(v.asnumpy(),(0,3,1,2))))})
            else:
                params_mx.update({k:mx.nd.array(np.float32(v.asnumpy()))})
        #mx.nd.save("/home/tusimple/Desktop/tvm_ir_test/conv2d_tensor_core/convf16/params/resnet50.params",params_mx)
  
        if gen_para:
            return(0)
        #create data
        data_np = np.random.uniform(0.,1.,size=input_shape).astype(dtype['data'])
        data_tvm = tvm.nd.array(data_np)
        data_mx = [mx.nd.array(np.float32(np.transpose(data_np,(0,3,1,2))))]   
        batch_mx=namedtuple("Batch",['data'])
        
        #run mxnet version of vgg net work
        #mvg.bind(data_shapes=[("data",(1,input_shape[3],256,256))])
        #mvg.set_params(params_mx,{})
        #mvg.forward(batch_mx(data_mx))

        #result_mx=mvg.get_outputs()[0]
        #result_mx_np=result_mx.asnumpy()
        
        #run tvm fp16 vgg network
        module = runtime.create(graph, lib, ctx)
        module.set_input('data', data_tvm)
        module.set_input(**params_tvm)

        module.run()
        result_tc = module.get_output(0)
        #print(result_tc.asnumpy())
        print("out_put get")
        #result_tc_np = result_tc.asnumpy()
        
        #verify the accuracy
        verify=False
        if verify:
            #np.testing.assert_allclose(result_tc_np,np.float16(result_mx_np),rtol=1e-2)
            print("verify success")
        else:
            print("accuracy not verified")

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
        return(0)


res=tune_and_evaluate(tuning_option)
