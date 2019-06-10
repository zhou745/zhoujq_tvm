"Example code to perform int8 GEMM"
import logging
import sys
import numpy as np
import tvm
from tvm import autotvm

DO_TUNING = True
PRETUNED_INDEX = 75333

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
        binds = {t: tvm.decl_buffer(t.shape, t.dtype, t.op.name,
                                    data_alignment=cfg.data_alignment,
                                    offset_factor=cfg.offset_factor,
                                    scope='local') for t in [x, y, z]}
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds=binds)


dot = intrin_dot()


@autotvm.template
def gemm_int8(n, m, l):
    A = tvm.placeholder((n, l), name='A', dtype='int8')
    B = tvm.placeholder((m, l), name='B', dtype='int8')

    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute((n, m), lambda i, j: tvm.sum(A[i, k].astype('int32') * B[j, k].astype(
        'int32'), axis=k), name='C')

    cfg = autotvm.get_config()
    s = tvm.create_schedule(C.op)
    y, x = C.op.axis

    AA = s.cache_read(A, 'shared', [C])
    BB = s.cache_read(B, 'shared', [C])
    AL = s.cache_read(AA, 'local', [C])
    BL = s.cache_read(BB, 'local', [C])
    CC = s.cache_write(C, 'local')

    k = CC.op.reduce_axis[0]

    cfg.define_split('tile_k', cfg.axis(k), num_outputs=3,
                     filter=lambda entity: entity.size[2] == 4 and \
                     entity.size[0] * 2 >= entity.size[1])

    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)

    s[CC].tensorize(ki, dot)

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')

    def block_size_filter(entity):
        return entity.size[0] * 2 >= entity.size[1] * 2 and \
                entity.size[1] <= 16 and entity.size[3] <= 4
    cfg.define_split('tile_y', cfg.axis(y), num_outputs=4, filter=block_size_filter)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=4, filter=block_size_filter)
    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, y)
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, x)

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tyz, tvm.thread_axis('vthread'))
    s[C].bind(txz, tvm.thread_axis('vthread'))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)

    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    s[CC].reorder(ko, kt, yo, xo, ki)
    s[CC].unroll(kt)

    for stage in [AL, BL]:
        s[stage].compute_at(s[CC], kt)
        _, xi = s[stage].split(stage.op.axis[1], factor=4)
        s[stage].vectorize(xi)
        s[stage].double_buffer()

    cfg.define_knob('storage_align', [16, 48])
    for stage in [AA, BB]:
        s[stage].storage_align(s[stage].op.axis[0],
                               cfg['storage_align'].val, 0)
        s[stage].compute_at(s[CC], ko)

        fused = s[stage].fuse(*s[stage].op.axis)
        ty, tx = s[stage].split(fused, nparts=cfg['tile_y'].size[2])
        tx, xi = s[stage].split(tx, nparts=cfg['tile_x'].size[2])
        _, xi = s[stage].split(xi, factor=16)

        s[stage].bind(ty, thread_y)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(xi)

    cfg.define_knob('auto_unroll_max_step', [512, 1500])
    s[C].pragma(by, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[C].pragma(by, 'unroll_explicit', False)

    cfg.add_flop(n*m*l*2)
    return s, [A, B, C]


if __name__ == '__main__':
    N = 2048
    n = m = l = N

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    task = autotvm.task.create(gemm_int8, args=(n, m, l), target='cuda')
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner(number=10, timeout=20))
    log_name = 'gemm_int8.log'
    if DO_TUNING:
        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(n_trial=1000, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])

        dispatch_context = autotvm.apply_history_best(log_name)
        best_config = dispatch_context.query(task.target, task.workload)
        print('\nBest config:')
        print(best_config)
    else:
        config = task.config_space.get(PRETUNED_INDEX)
        dispatch_context = autotvm.task.ApplyConfig(config)
        print("Using pretuned config:")
        print(config)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = gemm_int8(n, m, l)
            f = tvm.build(s, arg_bufs, 'cuda', name='gemm_int8')

    ctx = tvm.context('cuda', 0)

    a_np = np.random.randint(size=(n, l), low=-128, high=127, dtype='int8')
    b_np = np.random.randint(size=(m, l), low=-128, high=127, dtype='int8')

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, m), dtype='int32'), ctx)
    f(a, b, c)

    np.testing.assert_allclose(
        c.asnumpy(),
        np.dot(
            a_np.astype('int32'),
            b_np.T.astype('int32')),
        rtol=1e-5)

    num_ops = 2 * l * m * n
    num_runs = 1000
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GOPS = num_ops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GOPS." %
          (num_runs, t * 1e3, GOPS))