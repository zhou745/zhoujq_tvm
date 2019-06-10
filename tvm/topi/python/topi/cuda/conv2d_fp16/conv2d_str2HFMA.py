import functools
import tvm
from tvm import autotvm
import numpy as np
from ...nn.pad import pad
from ...nn.util import get_pad_tuple

def conv2d_NHWC_HFMA(cfg, data, kernel,data_shape,kernel_shape,output_shape,stride, padding):
    print("data size %s:"%data.name,data_shape)
    print("kernel size in layer %s:"%kernel.name,kernel_shape)
    print(output_shape)
    #input data and kernel
    data_batch,data_h,data_w,data_channel = data_shape
    kernel_num,kernel_h,kernel_w,kernel_channel=kernel_shape

    #pad the input data
    pad_before = [0,padding[0],padding[1],0]
    pad_after = pad_before

    data_pad = pad(data,pad_before,pad_after,name = "data_pad")

    #compute the conv
    rh = tvm.reduce_axis((0, kernel_h), name='rh')
    rw = tvm.reduce_axis((0, kernel_w), name='rw')

    rc = tvm.reduce_axis((0,data_channel),name='rc')

    conv = tvm.compute(output_shape,lambda dn,dp,dq,dk:tvm.sum(\
                       data_pad[dn][dp*stride[0]+rh][dq*stride[1]+rw][rc]*kernel[dk][rh][rw][rc],\
                       axis=[rh,rw,rc]),name = 'conv')

    O = tvm.compute(output_shape,lambda dn,dp,dq,dk: conv[dn][dp][dq][dk],name = "O",\
                    tag="conv2d_NHWC_HFMA",attrs={"version":2})
    
    num_flop = data_batch*output_shape[1]*output_shape[2]*kernel_num*data_channel*kernel_h*kernel_w*2
    cfg.add_flop(num_flop)
    return(O)

def schedule_conv2d_NHWC_HFMA(cfg,s,O):
    
    #get stages
    conv = O.op.input_tensors[0]
    data_pad,kernel=conv.op.input_tensors
    s[data_pad].compute_inline()

    #read from cache
    data_sh = s.cache_read(data_pad,'shared',[conv])
    kernel_sh = s.cache_read(kernel,'shared',[conv])

    s[conv].set_scope('local')
    if O.op not in s.outputs:
        s[O].compute_inline()
        O = s.outputs[0].output(0)
    #set tile cfg
    n,p,q,k = s[O].op.axis
    cfg.define_split('n_cut',cfg.axis(n),num_outputs=4)
    cfg.define_split('p_cut',cfg.axis(p),num_outputs=4)
    cfg.define_split('q_cut',cfg.axis(q),num_outputs=4)
    cfg.define_split('k_cut',cfg.axis(k),num_outputs=4)
    
    
    #apply axis cuttings
    kernel_scope, n = s[O].split(n, nparts=1)
    bn, vn, tn, ni = cfg["n_cut"].apply(s, O, n)
    bp, vp, tp, pi = cfg["p_cut"].apply(s, O, p)
    bq, vq, tq, qi = cfg["q_cut"].apply(s, O, q)
    bk, vk, tk, ki = cfg["k_cut"].apply(s, O, k)

    #reoder the computation order
    s[O].reorder(bn, bp, bq, bk, vn, vp, vq, vk, tn, tp, tq, tk, ni, pi, qi, ki)
    #s[O].vectorize(ki)
    #define thread and blocks
    bidx=tvm.thread_axis("blockIdx.x")
    bidy=tvm.thread_axis("blockIdx.y")
    bidz=tvm.thread_axis("blockIdx.z")

    tidx=tvm.thread_axis("threadIdx.x")
    tidy=tvm.thread_axis("threadIdx.y")
    tidz=tvm.thread_axis("threadIdx.z")

    vid1 = tvm.thread_axis("vthread",name="vid1")
    vid2 = tvm.thread_axis("vthread",name="vid2")
    vid3 = tvm.thread_axis("vthread",name="vid3")
    vid4 = tvm.thread_axis("vthread",name="vid4")

    #bind the output
    bpq = s[O].fuse(bp,bq)

    tpq = s[O].fuse(tp,tq)

    s[O].bind(bn,bidz)
    s[O].bind(bpq,bidy)
    s[O].bind(bk,bidx)

    s[O].bind(vn,vid1)
    s[O].bind(vp,vid2)
    s[O].bind(vq,vid3)
    s[O].bind(vk,vid4)

    s[O].bind(tn,tidz)
    s[O].bind(tpq,tidy)
    s[O].bind(tk,tidx)
    s[conv].compute_at(s[O],tk)
    
   
    #cut and bind conv
    n,p,q,k = s[conv].op.axis
    rr,rs,rc = s[conv].op.reduce_axis
    #rc,rct=s[conv].split(rc,factor=8)
    cfg.define_split("rr_cut", cfg.axis(rr), num_outputs=2)
    cfg.define_split("rs_cut", cfg.axis(rs), num_outputs=2)
    cfg.define_split("rc_cut", cfg.axis(rc), num_outputs=3)
    rro, rri = cfg['rr_cut'].apply(s, conv, rr)
    rso, rsi = cfg['rs_cut'].apply(s, conv, rs)
    rco, rci,rct = cfg['rc_cut'].apply(s, conv, rc)

    s[conv].reorder(rro, rso, rco, rri, rsi, rci,rct, n, p,q,k)
    
    #attach compute for read chache
    s[data_sh].compute_at(s[conv],rco)
    s[kernel_sh].compute_at(s[conv],rco)
    
    n_tz = cfg["n_cut"].size[2]
    n_ty = cfg["p_cut"].size[2]*cfg["q_cut"].size[2]
    n_tx = cfg["k_cut"].size[2]
    n_tt = cfg["rc_cut"].size[2]

    #bind the reading cache
    for load in [data_sh, kernel_sh]:
        
        ct= s[load].op.axis[-1]
        ct,tt=s[load].split(ct,factor=n_tt)
        s[load].vectorize(tt)
        fused = s[load].op.axis[:-1]+[ct]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, tidz)
        s[load].bind(ty, tidy)
        s[load].bind(tx, tidx)
    # double buffer

    cfg.define_knob('Data_sh_buf', [0, 1])
    cfg.define_knob('kernel_sh_buf', [0, 1])
    if cfg['Data_sh_buf'].val:
        s[data_sh].double_buffer()
    if cfg['kernel_sh_buf'].val:
        s[kernel_sh].double_buffer()

    #unroll
    #cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    #s[O].pragma(kernel_scope, 'auto_unroll_max_step',
    #               cfg['auto_unroll_max_step'].val)
    #s[O].pragma(kernel_scope, 'unroll_explicit', False)
    return(s)