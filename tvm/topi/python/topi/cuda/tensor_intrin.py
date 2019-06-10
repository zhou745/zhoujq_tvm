"""Tensor intrinsics on CUDA."""
#pylint: disable=invalid-name
import tvm


def dp4a(x_scope='local', y_scope='local', z_scope='local'):
    """
    Int8 dot product reduced by every 4 elements using __dp4a

    Parameters
    ----------
    x_scope : str, optional
        The storage scope of buffer for lhs
    y_scope : str, optional
        The storage scope of buffer for rhs
    z_scope : str, optional
        The storage scope of buffer for result

    Returns
    -------
    intrin : TensorIntrin
        The dp4a TensorIntrin that can be used in tensorizing schedule.
    """

    n = 4  # dp4a requires operands packed by 4
    x = tvm.placeholder((n,), name='x', dtype='int8')
    y = tvm.placeholder((n,), name='y', dtype='int8')

    k = tvm.reduce_axis((0, n), name='rc')

    z = tvm.compute((1,), lambda i: tvm.sum(
        x[k].astype('int32') * y[k].astype('int32'), axis=[k]))

    def _intrin_func(ins, outs):
        def _instr(index):
            xx, yy = ins
            zz = outs[0]

            if index == 1:
                return zz.vstore(0, 0)

            ib = tvm.ir_builder.create()

            vec_x = xx.vload(0, dtype='int8x4')
            vec_y = yy.vload(0, dtype='int8x4')
            prev_z = 0 if index == 0 else zz.vload(0)

            new_z = tvm.call_pure_extern('int32', '__dp4a', vec_x, vec_y, prev_z)
            ib.emit(zz.vstore(0, new_z))

            return ib.get()

        return _instr(0), _instr(1), _instr(2) # body, reset, update

    with tvm.build_config(data_alignment=4, offset_factor=1) as cfg:
        scopes = {x: x_scope, y: y_scope, z: z_scope}
        binds = {t: tvm.decl_buffer(t.shape, t.dtype, t.op.name,
                                    data_alignment=cfg.data_alignment,
                                    offset_factor=cfg.offset_factor,
                                    scope=scopes[t]) for t in [x, y, z]}

        return tvm.decl_tensor_intrin(z.op, _intrin_func, binds=binds)


def int4_copy(x_scope = "global",y_scope = "global",add_on=None,bidx=tvm.thread_axis('blockIdx.x'),npq=1024,blk_size=64):
    n = 8  # int4_copy requires operands packed by 8 for fp16
    x = tvm.placeholder((n,), name='x', dtype='float16')
    y = tvm.compute((n,), lambda i: x[i])
    
    xb = tvm.decl_buffer(x.shape,x.dtype,name="x",scope=x_scope,offset_factor=1)
    yb = tvm.decl_buffer(y.shape,y.dtype,name="y",scope=y_scope,offset_factor=1)
    offset_x = -(blk_size)*(bidx/npq)
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        if(add_on == None):
            int4copy = tvm.call_intrin('float32', 'int4_copy', yb.access_ptr("w"),0,xb.access_ptr("r"),offset_x)
            ib.emit(int4copy)
        elif(add_on == "relu"):
            relu = tvm.call_intrin('float32','relu',xb.access_ptr("w"),0)
            ib.emit(relu)
            int4copy = tvm.call_intrin('float32', 'int4_copy', yb.access_ptr("w"),0,xb.access_ptr("r"),offset_x)
            ib.emit(int4copy)
        return ib.get()

    with tvm.build_config() as cfg:
        return tvm.decl_tensor_intrin(y.op, intrin_func, binds={x:xb,y:yb})