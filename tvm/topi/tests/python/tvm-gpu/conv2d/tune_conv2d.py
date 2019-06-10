import sys
import logging
import os
import argparse
import tvm
from tvm import autotvm
import topi
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('B', type=int)
parser.add_argument('CI', type=int)
parser.add_argument('size', type=int)
parser.add_argument('CO', type=int)
parser.add_argument('kernel_size', type=int)
parser.add_argument('stride', type=int)
parser.add_argument('padding', type=int)
parser.add_argument('dilation', type=int)
parser.add_argument('groups', type=int)

parser.add_argument('--n_trial', type=int, default=500)
parser.add_argument('--log_file', type=str, default='group_conv2d.log')
parser.add_argument('--bench_only', action='store_true')
parser.add_argument('--append', action='store_true')


args = parser.parse_args()
print(args)

ic_block = 4
oc_block = 4

target = tvm.target.cuda()

tuning_option = {
    'n_trial': args.n_trial,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=10, repeat=3, timeout=3),
    )
}


autotvm.task.nnvm_integration.TaskExtractEnv.get()

ctx = tvm.gpu()


if args.groups > 1:
    compute_func = topi.nn.group_conv2d_nchw
    schedule_func = topi.generic.schedule_group_conv2d_nchw
    topi_name = 'topi_nn_group_conv2d_nchw'
else:
    compute_func = topi.nn.conv2d
    schedule_func = topi.generic.schedule_conv2d_nchw
    topi_name = 'topi_nn_conv2d'


def get_original_args():
    data = tvm.placeholder((args.B, args.CI, args.size, args.size), dtype='int8')
    kernel = tvm.placeholder((args.CO, args.CI // args.groups, args.kernel_size, args.kernel_size), dtype='int8')

    if args.groups > 1:
        _args = (data, kernel, args.stride, args.padding, args.dilation, args.groups, 'int8')
    else:
        _args = (data, kernel, args.stride, args.padding, args.dilation, 'NCHW', 'int8')
    return _args


def get_transformed_args():
    data = tvm.placeholder((args.B, args.CI//ic_block, args.size, args.size, ic_block), dtype='int8')
    kernel = tvm.placeholder((args.CO // oc_block, args.CI // ic_block // args.groups, args.kernel_size, args.kernel_size, oc_block, ic_block), dtype='int8')

    if args.groups > 1:
        _args = (data, kernel, args.stride, args.padding, args.dilation, args.groups, 'int8')
    else:
        _args = (data, kernel, args.stride, args.padding, args.dilation, 'NCHW', 'int8')
    return _args


def create_task():
    _args = get_original_args()
    _args = autotvm.task.nnvm_integration.serialize_args(_args)
    task = autotvm.task.create(topi_name, _args, target=target, template_key='int8')

    return task


def tune_task(task,
               measure_option,
               n_trial=1000,
               early_stopping=None,
               ):
    # do tuning
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=n_trial,
                   early_stopping=early_stopping,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(args.log_file)])


def update():
    workload = autotvm.task.args_to_workload(get_original_args(), compute_func)
    cfg = autotvm.DispatchContext.current.query(target, workload)

    if cfg.template_key == 'int8':
        new_workload = autotvm.task.args_to_workload(get_transformed_args(), compute_func)
        autotvm.DispatchContext.current.update(target, new_workload, cfg)


def _bench():
    _args = get_transformed_args()
    output = compute_func(*_args)
    s = schedule_func(output)
    f = tvm.build(s, [_args[0], _args[1], output])
    #print(f)
    #print(f.imported_modules[0].get_source())
    evaluator = f.time_evaluator(f.entry_name, ctx, number=1, repeat=1)
    a_np = np.random.randint(size=(args.B, args.CI//ic_block, args.size, args.size, ic_block), low=-128, high=127, dtype='int8')
    w_np = np.random.randint(size=(args.CO // oc_block, args.CI // ic_block // args.groups, args.kernel_size, args.kernel_size, oc_block, ic_block), low=-128, high=127, dtype='int8')
    a_tvm = tvm.nd.empty(a_np.shape, dtype='int8', ctx=ctx).copyfrom(a_np)
    w_tvm = tvm.nd.empty(w_np.shape, dtype='int8', ctx=ctx).copyfrom(w_np)
    osize = (args.padding * 2 + args.size - (args.kernel_size-1)*args.dilation - 1) // args.stride + 1
    c_tvm = tvm.nd.empty((args.B, args.CO // oc_block, osize, osize, oc_block), ctx=ctx, dtype='int8')
    f(a_tvm, w_tvm, c_tvm)
    f(a_tvm, w_tvm, c_tvm)
    t = evaluator(a_tvm, w_tvm, c_tvm).mean
    num_flop = args.B * osize * osize * args.CO * args.CI * args.kernel_size * args.kernel_size * 2 // args.groups
    GFLOPS = num_flop / (t * 1e3) / 1e6
    print('Time cost of this operator: %f, %g GFLOPS' % (t, GFLOPS))


def bench():
    with autotvm.apply_history_best(args.log_file):
        with target:
            update()
            _bench()

def main():
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    if not args.bench_only:
        if not args.append and os.path.exists(args.log_file):
            os.remove(args.log_file)

        task = create_task()
        tune_task(task, **tuning_option)

    bench()


if __name__ == '__main__':
    main()
