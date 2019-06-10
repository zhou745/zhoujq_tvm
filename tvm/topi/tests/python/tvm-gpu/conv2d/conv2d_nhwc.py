import sys
import logging
import tvm
from tvm import autotvm
import topi
from topi.nn.pad import pad
import numpy as np
from topi.testing import conv2d_nhwc_python

DO_TUNE = False
PRETUNED_INDEX = 278521


def intrin_dot():
    n = 4  # dp4a requires operands packed by 4
    x = tvm.placeholder((n,), name='x', dtype='int8')
    y = tvm.placeholder((n,), name='y', dtype='int8')
    k = tvm.reduce_axis((0, n), name='k')

    z = tvm.compute(
        (1,), lambda _: tvm.sum(
            x[k].astype('int32') * y[k].astype('int32'), axis=k))

    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
        ib = tvm.ir_builder.create()

        dp4a = zz.vstore(0, tvm.call_pure_extern('int32', '__dp4a',
                                                 xx.vload(0, dtype='int8x4'),
                                                 yy.vload(0, dtype='int8x4'),
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


@autotvm.template
def conv2d(N, H, W, CI, CO, KH, KW, stride, padding):
    data = tvm.placeholder((N, H, W, CI), name='data', dtype='int8')
    kernel = tvm.placeholder((CO, KH, KW, CI), name='kernel', dtype='int8')

    HPAD, WPAD = padding
    HSTR, WSTR = stride

    pad_height = H + 2 * HPAD
    pad_width = W + 2 * WPAD

    out_height = (pad_height - KH) // HSTR + 1
    out_width = (pad_width - KW) // WSTR + 1

    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        pad_data = pad(data, (0, 0, HPAD, WPAD), name='pad_data')
    else:
        pad_data = data

    rc = tvm.reduce_axis((0, CI), name='rc')
    ry = tvm.reduce_axis((0, KH), name='ry')
    rx = tvm.reduce_axis((0, KW), name='rx')

    output = tvm.compute(
        (N, out_height, out_width, CO),
        lambda nn, yy, xx, ff: tvm.sum(
            pad_data[nn, yy * HSTR + ry, xx * WSTR + rx, rc].astype('int32') *
            kernel[ff, ry, rx, rc].astype('int32'), axis=[ry, rx, rc])
    )

    s = tvm.create_schedule([output.op])

    # inline padding
    if DOPAD:
        s[pad_data].compute_inline()

    data, raw_data = pad_data, data

    OL = s.cache_write(output, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, y, x, f = s[output].op.axis
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
    cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(by, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.z"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile and bind reduction axes
    n, f, y, x = s[OL].op.axis
    ry, rx, rc = s[OL].op.reduce_axis
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=3)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=3)
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3,
                     filter=lambda entity: entity.size[2] == 4)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    s[OL].reorder(ryo, rxo, rco, rym, rxm, rcm, n, f, y, x, ryi, rxi, rci)

    s[AA].compute_at(s[OL], rco)
    s[WW].compute_at(s[OL], rco)

    s[OL].tensorize(rci, dot)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        fused, tx = s[load].split(fused, factor=cfg["tile_f"].size[2])
        fused, ty = s[load].split(fused, factor=cfg["tile_x"].size[2])
        fused, tz = s[load].split(fused, factor=cfg["tile_y"].size[2])
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))

    # tune unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', False)

    # num flop
    NH, NW = [e.value for e in output.shape[1:3]]
    cfg.add_flop(N*CO*NH*NW*(CI*KH*KW*2))
    return s, [raw_data, kernel, output]


# logging config (for printing tuning log to screen)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (
    1, 1), (1, 1)
task = autotvm.task.create(conv2d,
                           args=(N, H, W, CO, CI, KH, KW, strides, padding),
                           target='cuda')
print(task.config_space)

if DO_TUNE:
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
                                            runner=autotvm.LocalRunner(number=10, timeout=4))

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=200,
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

# apply history best from log file
with dispatch_context:
    with tvm.target.create("cuda"):
        s, arg_bufs = conv2d(
            N, H, W, CO, CI, KH, KW, strides, padding)
        print(tvm.lower(s, arg_bufs, simple_mode=True))
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.randint(size=(N, H, W, CI), low=-128, high=127)
w_np = np.random.randint(size=(KH, KW, CI, CO), low=-128, high=127)
c_np = conv2d_nhwc_python(a_np, w_np, strides, padding)
w_np = w_np.transpose(3, 0, 1, 2)

ctx = tvm.gpu()
a_tvm = tvm.nd.empty(a_np.shape, dtype='int8', ctx=ctx).copyfrom(a_np)
w_tvm = tvm.nd.empty(w_np.shape, dtype='int8', ctx=ctx).copyfrom(w_np)
c_tvm = tvm.nd.empty(c_np.shape, dtype='int32', ctx=ctx)
func(a_tvm, w_tvm, c_tvm)

#np.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

# Evaluate running time. Here we choose a large repeat number (200) to reduce the noise
# and the overhead of kernel launch. You can also use nvprof to validate the result.

evaluator = func.time_evaluator(func.entry_name, ctx, number=5)
t = evaluator(a_tvm, w_tvm, c_tvm).mean
num_flops = N*c_np.shape[-2] * c_np.shape[-3] * CO*CI*KH*KW*2
GFLOPS = num_flops / (t * 1e3) / 1e6
print('Time cost of this operator: %f, %g GFLOPS' % (t, GFLOPS))
