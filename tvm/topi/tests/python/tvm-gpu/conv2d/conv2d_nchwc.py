import sys
import logging
import tvm
from tvm import autotvm
import topi
import numpy as np
from topi.testing import conv2d_nchw_python
from topi.nn.pad import pad
import functools
import operator
import os

DO_TUNING = True
PRETUNED_INDEX = 54961

def intrin_dot():
    a = 4  # dp4a requires operands packed by 4
    x = tvm.placeholder((4,), name='x', dtype='int8')
    y = tvm.placeholder((4,), name='y', dtype='int8')

    k = tvm.reduce_axis((0, 4), name='rc')

    z = tvm.compute((1,), lambda i: tvm.sum(
        x[k].astype('int32') * y[k].astype('int32'), axis=[k]))

    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
        ib = tvm.ir_builder.create()

        dp4a = zz.vstore(0, tvm.call_pure_extern('int32', '__dp4a',
                                                 xx.vload(
                                                     0, dtype='int8x4'),
                                                 yy.vload(
                                                     0, dtype='int8x4'),
                                                 zz.vload(0)))

        ib.emit(dp4a)

        body = ib.get()

        return body, zz.vstore(0, 0), body

    with tvm.build_config(data_alignment=4, offset_factor=1) as cfg:
        scopes = {x: 'shared', y: 'shared', z: 'local'}
        binds = {t: tvm.decl_buffer(t.shape, t.dtype, t.op.name,
                                    data_alignment=cfg.data_alignment, offset_factor=cfg.offset_factor,
                                    scope=scopes[t]) for t in [x, y, z]}
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds=binds)


dot = intrin_dot()

BI = BO = 4


@autotvm.template
def conv2d(N, H, W, CI, CO, KH, KW, strides, padding, scaling_factor):
    cfg = autotvm.get_config()

    data = tvm.placeholder((N, CI / BI, H, W, BI), name='data', dtype='int8')
    kernel = tvm.placeholder(
        (CO / BO, CI / BI, KH, KW, BO, BI), name='kernel', dtype='int8')

    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    stride_h, stride_w = (strides, strides) if isinstance(
        strides, int) else stride

    pad_height = H + 2 * pad_h
    pad_width = W + 2 * pad_w

    out_height = (pad_height - KH) // stride_h + 1
    out_width = (pad_width - KW) // stride_w + 1

    DOPAD = (stride_h != 0 or stride_w != 0)
    if DOPAD:
        pad_data = pad(data, (0, 0, pad_h, pad_w, 0), name='pad_data')
    else:
        pad_data = data

    oshape = (N, CO / BO, out_height, out_width, BO)

    ic_chunk = tvm.reduce_axis((0, CI/BI), name='ic_chunk')
    ic_block = tvm.reduce_axis((0, BI), name='ic_block')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(pad_data[n, ic_chunk, oh*stride_h+kh, ow*stride_w+kw, ic_block]
                               .astype('int32') *
                               kernel[oc_chunk, ic_chunk,
                                      kh, kw, oc_block, ic_block]
                               .astype('int32'),
                               axis=[ic_chunk, kh, kw, ic_block]))

    output = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
            (conv[n, oc_chunk, oh, ow, oc_block]*scaling_factor).astype('int8'), name='conv')

    s = tvm.create_schedule([output.op])
    s[conv].set_scope('local')

    # inline padding
    if DOPAD:
        s[pad_data].compute_inline()

    data, raw_data = pad_data, data

    # create cache stage
    AA = s.cache_read(data, 'shared', [conv])
    WW = s.cache_read(kernel, 'shared', [conv])

    # tile and bind spatial axes
    n, f, y, x, c = s[output].op.axis
    cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    s[output].bind(n, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    _, c = s[output].split(c, factor=4)
    #s[output].vectorize(c)

    s[conv].compute_at(s[output], tx)

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis

    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg['tile_rc'].apply(s, conv, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, conv, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)

    _, rc_block = s[conv].split(rc_block, factor=4)
    s[conv].tensorize(rc_block, dot)

    s[AA].compute_at(s[conv], rxo)
    s[WW].compute_at(s[conv], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        if load == AA:
            n, f, y, x, c = s[load].op.axis
            if not DOPAD:
                s[load].vectorize(c)
                fused = s[load].fuse(n, f, y, x)
            else:
                c, _ = s[load].split(c, factor=4)
                fused = s[load].fuse(n, f, y, x, c)
        else:
            n, f, y, x, oc_chunk, c = s[load].op.axis
            fused = s[load].fuse(n, f, y, x, oc_chunk)
            s[load].vectorize(c)

        fused, tx = s[load].split(fused, factor=cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, factor=cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, factor=cfg["tile_f"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    for load in [AA, WW]:
        name = load.op.name + '_double_buffer'
        cfg.define_knob(name, [0, 1])

        if cfg[name].val:
            s[load].double_buffer

    # tune unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', False)

    # num flop
    NH, NW = [e.value for e in output.shape[2:4]]
    cfg.add_flop(N*CO*NH*NW*(CI*KH*KW*2))
    return s, [raw_data, kernel, output]


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

N, H, W, CO, CI, KH, KW, strides, padding, scaling_factor = 1, 14, 14, 512, 512, 1, 1, 2, 0, 1.0
task = autotvm.task.create(conv2d,
                           args=(N, H, W, CO, CI, KH, KW, strides, padding, scaling_factor),
                           target='cuda')

measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
                                        runner=autotvm.LocalRunner(number=10, timeout=4))

if DO_TUNING:
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=2000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('conv2d.log')])

    dispatch_context = autotvm.apply_history_best("conv2d.log")
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
else:
    config = task.config_space.get(PRETUNED_INDEX)
    dispatch_context = autotvm.task.ApplyConfig(config)
    print("Using pretuned config:")
    print(config)

with dispatch_context:
    with tvm.target.create("cuda"):
        s, arg_bufs = conv2d(
            N, H, W, CO, CI, KH, KW, strides, padding, scaling_factor)
        print(tvm.lower(s, arg_bufs, simple_mode=True))
        func = tvm.build(s, arg_bufs)
        print(func.imported_modules[0].get_source())

# check correctness
a_np = np.random.randint(size=(N, CI//BI, H, W, BI), low=-128, high=127, dtype='int8')
w_np = np.random.randint(
    size=(CO//BO, CI//BI, KH, KW, BO, BI), low=-128, high=127, dtype='int8')
a_np_ = a_np.transpose((0, 1, 4, 2, 3)).ravel().reshape(N, CI, H, W)
w_np_ = w_np.transpose((0, 4, 1, 5, 2, 3)).ravel().reshape(CO, CI, KH, KW)
c_np = conv2d_nchw_python(a_np_, w_np_, strides, padding).astype('int8')
c_np = c_np.reshape(N, CO//BO, BO, *c_np.shape[2:]).transpose(0, 1, 3, 4, 2)

ctx = tvm.gpu()
a_tvm = tvm.nd.empty(a_np.shape, dtype='int8', ctx=ctx).copyfrom(a_np)
w_tvm = tvm.nd.empty(w_np.shape, dtype='int8', ctx=ctx).copyfrom(w_np)
c_tvm = tvm.nd.empty(c_np.shape, dtype='int8', ctx=ctx)
func(a_tvm, w_tvm, c_tvm)

np.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
t = evaluator(a_tvm, w_tvm, c_tvm).mean
num_flops = N*c_np.shape[-2] * c_np.shape[-3] * CO*CI*KH*KW*2
GFLOPS = num_flops / (t * 1e3) / 1e6
print('Time cost of this operator: %f, %g GFLOPS' % (t, GFLOPS))
