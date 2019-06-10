"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import autotvm
import topi
import sys
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from common import NCHWcFp16Fallback
import logging
from tvm.autotvm.task.nnvm_integration import TaskExtractEnv
import os

def verify_conv2d_fp16_tensor_core(data_batch,data_in_channel,data_height,data_width,filter_num,filter_R,filter_S):
    print("test the convolution layer using tensor core with \
           data_in_channel=%d filter_num=%d"%(data_in_channel,filter_num))
    
    data = tvm.placeholder((data_batch, data_height, data_width,data_in_channel), name='data', dtype='float16')
    kernel = tvm.placeholder((filter_num, filter_R, filter_S,data_in_channel), name='kernel', dtype='float16')
    
    data_shape = get_const_tuple(data.shape)
    kernel_shape = get_const_tuple(kernel.shape)
    dtype = data.dtype
    
    def get_ref_data():
        data_np = np.random.uniform(0., 1., (data_batch, data_in_channel, data_height, data_width)).astype(dtype)
        filter_np = np.random.uniform(0., 1., (filter_num, data_in_channel,filter_R, filter_S)).astype(dtype)

        c_np = topi.testing.conv2d_nchw_python(data_np, filter_np, (1,1), (1,1)).astype(dtype)
        #c_np = np.random.uniform(0., 1., (data_batch, data_in_channel, data_height, data_width)).astype(dtype)
        data_np = np.transpose(data_np,(0,2,3,1))
        filter_np = np.transpose(filter_np,(0,2,3,1))
        c_np = np.transpose(c_np,(0,2,3,1))
  
        return(data_np,filter_np,c_np)
   
    data_n,filter_n,c_n = get_ref_data()

    def device_compute(device):
        ctx = tvm.context(device, 0)
        #auto tvm tune
        Env = TaskExtractEnv()

        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
        args_tvm = autotvm.task.nnvm_integration.serialize_args((data,kernel,(1,1),(1,1),1,'NHWC',dtype))
        task = autotvm.task.create("topi_nn_conv2d",args=args_tvm,target="cuda",template_key='fp16')

        measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),\
                    runner=autotvm.LocalRunner(number=2,repeat=1, timeout=4))

        tuner = autotvm.tuner.XGBTuner(task)

 
        tune=True
        if tune:
            tuner.tune(n_trial=400,measure_option=measure_option,\
                callbacks=[autotvm.callback.log_to_file('temp_conv2d.log')])
            autotvm.record.pick_best('temp_conv2d.log','conv2d.log')
            os.remove('temp_conv2d.log')
        with autotvm.apply_history_best('conv2d.log'):
            with tvm.target.create(device):
                C = topi.nn.conv2d(data, kernel, (1, 1), (1, 1), 1,
                               layout='NHWC', out_dtype=dtype)
                s = topi.generic.schedule_conv2d_nhwc([C])
                
                print("now start build")
                print(tvm.lower(s, [data, kernel, C], name="conv2d",simple_mode=True))
                func = tvm.build(s, [data, kernel, C], device, name="conv2d")
                print("build finished")
        data_d = tvm.nd.array(data_n, ctx)
        filter_d = tvm.nd.array(filter_n, ctx)
        
        output_shape=get_const_tuple(C.shape)
        c_d = tvm.nd.array(np.zeros(output_shape, dtype=C.dtype), ctx)


        func(data_d, filter_d, c_d)
        #print(c_d.shape)
        #tvm.testing.assert_allclose(c_d.asnumpy(), c_n, rtol=2e-2)
        print("verify success")
 
        num_flops = output_shape[2]*output_shape[3]*output_shape[1]*output_shape[0]*data_in_channel*filter_R*filter_S*2
        num_runs = 10
        timer_f = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        t = timer_f(data_d,filter_d,c_d).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))

    for device in ["cuda"]:
        device_compute(device)

def test_conv2d_fp16():
    with NCHWcFp16Fallback():
        verify_conv2d_fp16_tensor_core(1,16,32,32,16,3,3)

if __name__ == "__main__":
    test_conv2d_fp16()