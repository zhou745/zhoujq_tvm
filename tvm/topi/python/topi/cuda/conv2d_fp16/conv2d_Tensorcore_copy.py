import functools
import tvm
from tvm import autotvm
import numpy as np
from ...nn.pad import pad
from ...nn.util import get_pad_tuple


def conv2d_Tensor_core_copy(cfg,data,data_shape):
    data_batch,data_h,data_w,data_channel = data_shape
    O = tvm.compute(data_shape,lambda dn,dp,dq,dk: data[dn][dp][dq][dk],name = "O",\
                    tag="conv2d_Tensor_core_copy",attrs={"version":3})
    num_flop = data_batch*data_channel*data_h*data_w
    cfg.add_flop(num_flop)
    return(O)
   
def schedule_conv2d_Tensor_core_copy(cfg,s,O):
    #get stages
    data = O.op.input_tensors[0]

    #read from cache
    data_local = s.cache_read(data,'local',[O])

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
    
    
    #attach compute for read chache
    s[data_local].compute_at(s[O],ki)
    

    # double buffer

    cfg.define_knob('Data_local_buf', [0, 1])
    if cfg['Data_local_buf'].val:
        s[data_local].double_buffer()

    return(s)