import tvm
import numpy as np
from tvm import autotvm
from ..util import get_const_tuple
from tensor_intrin import int4_copy


def compute_outshape(data_batch,data_height,data_width,filter_num,filter_R,filter_S,pad=[1,1],stride=[1,1]):
    """
    This function is used for computing the shape for the out put of current kernel   
    """

    output_height = int(np.ceil(float(data_height-filter_R+1+2*pad[0])/float(stride[0])))
    output_width = int(np.ceil(float(data_width-filter_S+1+2*pad[1])/float(stride[1])))
    outshape = (data_batch,output_height,output_width,filter_num)

    return(outshape)
bidx=tvm.thread_axis('blockIdx.x')   
tidx=tvm.thread_axis('threadIdx.x')

def conv2d_fp16_tensor_core(cfg, data, kernel):
    """The data layout should be NCHW, while the kernel layout is KRSC 
    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, filter_height, filter_width, in_channel]

    currently only pad = 1, stride =1 output type =fp16 is enabled, should be expanded later

    """
    #get size for both data and kernels
    data_batch, data_height, data_width,data_in_channel = get_const_tuple(data.shape)
    filter_num,filter_R, filter_S,filter_in_channel = get_const_tuple(kernel.shape)
    #print(data_batch, data_height, data_width,data_in_channel)
    #print(filter_num,filter_R, filter_S,filter_in_channel)
    #compute the shape for output
    output_shape = compute_outshape(data_batch,data_height,data_width,filter_num,filter_R,filter_S)
    print(output_shape)
    if(output_shape[1]>0):
        fortype="unroll"
    else:
        fortype="serial"

    #kernel parameter
    #block_para
    blk_q,block_row_warp,warp_col_tile,compute_type=choose_best_tiling_para(output_shape[2],output_shape[3])
    blk_p = 8
    blk_size = blk_p*blk_q
    
    ko_part = 2
    ki_part = output_shape[3]/16
    #tiling parameters
    block_col_warp = 2
    warp_row_tile = 2
    #offset preset
    shieft =8
    offset_D_im2col = (2+blk_q)*(2+blk_p)*16
    offset_F = offset_D_im2col+(shieft+16)*blk_size
 
    npq = output_shape[1]*output_shape[2]/blk_size
    #GPU information
    num_sm=80
    output_copy = blk_size*blk_size
    compute = offset_D_im2col+(shieft+16)*blk_size*2

    shmem_use = max(output_copy,compute)
    def convolutionfp16(D,F,shmem):
        #ir builder for constructing the main body
        ib = tvm.ir_builder.create()
        
        #compute the current block number
        blk_id = bidx/num_sm
    
        #id of current warp and offset of shared memory when storing
        warpid=tidx/32
        warp_offset_output = warpid%block_row_warp*16*warp_row_tile\
            +warpid/block_row_warp*warp_col_tile*block_row_warp*warp_row_tile*256

        #include necessary head files 
        include_file=tvm.call_intrin("float32","include_cpp_head","/home/tusimple/Desktop/tvm_ir_test/conv2d_tensor_core/conv2dv9.h")
        ib.emit(include_file)

        #declare the matrix fragment
        declare_a = tvm.call_intrin("float32","wmma_fragment","matrix_a","half","row_major","a_frag",warp_col_tile)
        declare_b = tvm.call_intrin("float32","wmma_fragment","matrix_b","half","col_major","b_frag",warp_row_tile)
        declare_c = tvm.call_intrin("float32","wmma_fragment","accumulator","half","c_frag",warp_col_tile,warp_row_tile)
        ib.emit(declare_a)
        ib.emit(declare_b)
        ib.emit(declare_c)

        #define the shared memory for loading data and offset for loading the data
        offset_D_warp = offset_D_im2col+tidx/2*(16+shieft)+tidx%2*8
        offset_F_warp = offset_F+tidx/2*(16+shieft)+tidx%2*8

        #ir template for thread synchronization 
        sync = tvm.call_extern("float32","__syncthreads")

        #main for conducting the computation
        #set the pointer to first address of D
        Dp=D.access_ptr("r")
        Sp=shmem.access_ptr("r")
        Fp=F.access_ptr("r")

        #load the first data from global memory for the reuse of 9 times
        load_first_data = tvm.call_extern("float32","load_matrix_D",Dp,Sp,blk_id,\
                                            data_batch,data_height,data_width,data_in_channel,0,compute_type)
        ib.emit(load_first_data)
        #set the pointer to beginning of F
        Fp=F.access_ptr("r")

        #load the first filter from global memory:
        load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,blk_id,filter_num,\
                                    filter_in_channel,data_batch,data_height,data_width,tidx%2*8,0,compute_type)
        ib.emit(load_filter)
            
        #fill fragment c with 0
        with ib.for_range(0,warp_col_tile,name = "col_id_fi") as col_id_fi:
            with ib.for_range(0,warp_row_tile, name = "row_id_fi") as row_id_fi:              
                fill_O_zero = tvm.call_intrin("float","wmma_fill_fragment","c_frag",col_id_fi,row_id_fi,"half",0.)
                ib.emit(fill_O_zero)
        ib.emit(sync)

        #do im2col for the first data
        im2col=tvm.call_extern("float32","im2col",Sp,offset_D_warp,0,compute_type)
        ib.emit(im2col)
        ib.emit(sync)
                
        with ib.for_range(0,data_in_channel/16,name="c_id",for_type=fortype) as c_id:
            with ib.for_range(0,9,name="ker_id",for_type=fortype) as ker_id:
                #now load matrix fragment
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    load_matrix_frag_F = tvm.call_intrin("float32","wmma_load_matrix_sync","a_frag",col,Sp,\
                                                        offset_D_im2col+tidx/(32*block_row_warp)*\
                                                        (16*warp_col_tile*(16+shieft))+col*(16*(16+shieft)),16+shieft)
                    ib.emit(load_matrix_frag_F)
        
                with ib.for_range(0,warp_row_tile,name = "row") as row:
                    load_matrix_frag_D = tvm.call_intrin("float32","wmma_load_matrix_sync","b_frag",row,Sp,\
                                                        offset_F+tidx%(32*block_row_warp)/32*\
                                                        (16*warp_row_tile*(16+shieft))+row*(16*(16+shieft)),16+shieft)
                    ib.emit(load_matrix_frag_D)
                ib.emit(sync)
                #now compute
                with ib.for_range(0,warp_col_tile,name = "mma_col") as mma_col:
                    with ib.for_range(0,warp_row_tile,name = "mma_row") as mma_row:
                        wmma_compute = tvm.call_intrin("float16","wmma_mma_sync","c_frag","a_frag","b_frag","c_frag",mma_col,mma_row)
                        ib.emit(wmma_compute)
            
                with ib.if_scope(ker_id<8):
                    #load filer of the next ieration
                    load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,blk_id,filter_num,filter_in_channel,\
                                                data_batch,data_height,data_width,c_id*16+tidx%2*8,ker_id+1,compute_type)
                    ib.emit(load_filter)
                    #load data for next iteration
                    im2col=tvm.call_extern("float32","im2col",Sp,offset_D_warp,ker_id+1,compute_type)
                    ib.emit(im2col)
                ib.emit(sync)

            with ib.if_scope(c_id<data_in_channel/16-1):
                #load the next 9 iteration data from global memory
                load_data = tvm.call_extern("float32","load_matrix_D",Dp,Sp,blk_id,\
                                data_batch,data_height,data_width,data_in_channel,c_id*16+16,compute_type)
                ib.emit(load_data)

                #load filter for next cd iter
                load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,blk_id,filter_num,\
                                            data_in_channel,data_batch,data_height,data_width,c_id*16+16+tidx%2*8,0,compute_type)
                ib.emit(load_filter)
                ib.emit(sync)

                #load the first data from shmem to im2col shmem
                im2col=tvm.call_extern("float32","im2col",Sp,offset_D_warp,0,compute_type)
                ib.emit(im2col)
                ib.emit(sync)

        #store fragment in shared memory first
        with ib.for_range(0,warp_col_tile,name = "col_id_st") as col_id_st:
            with ib.for_range(0,warp_row_tile, name = "row_id_st") as row_id_st:
                store_O_fragment = tvm.call_intrin("float32","wmma_store_matrix_sync",Sp,warp_offset_output+col_id_st*(256*warp_row_tile*block_row_warp)+row_id_st*16,"c_frag",col_id_st,row_id_st,64)
                ib.emit(store_O_fragment)
        ib.emit(sync)

        body = ib.get()
        return(body)


    shmem = tvm.extern((shmem_use,),[data,kernel],lambda ins,outs:convolutionfp16(ins[0],ins[1],outs[0]),\
                        name = "shmem",dtype = 'float16',\
                        out_buffers=tvm.decl_buffer((shmem_use,),dtype='float16',scope='shared',offset_factor=1))
    O = tvm.compute(output_shape,lambda dn,dp,dq,dk:shmem[dk+dq%blk_q*blk_size+dp%blk_p*blk_size*blk_q],tag="conv2d_fp16_tensor_core",\
                    attrs={"blk_size":blk_size,"npq":npq})
    
    num_flop =  data_batch* output_shape[2]*output_shape[3]*filter_num*2*data_in_channel*filter_R*filter_S
    cfg.add_flop(num_flop)
    
    return(O)


def schedule_conv2d_fp16_tensor_core(cfg,s,O):
    blk_p=8
    ko_part=2
    ki_factor=8

    npq=O.op.attrs["npq"]
    blk_size=O.op.attrs["blk_size"]
    blk_q = blk_size/blk_p

    print("npq=%d,blk_size=%d"%(npq,blk_size))
    _int4_copy = int4_copy(x_scope="shared",y_scope="global",add_on=None,bidx=bidx,npq=npq,blk_size=blk_size)
    shmem = O.op.input_tensors[0]
    n,p,q,k = s[O].op.axis
    
   
    #split the axis for computation
    bp,pi = s[O].split(p,factor = blk_p)
    bq,qi = s[O].split(q,factor = blk_q)
    bk,ki = s[O].split(k,factor = blk_size)
    ko,ki = s[O].split(ki,nparts = ko_part)
    ki,kt = s[O].split(ki,factor = ki_factor)

    s[O].reorder(n,bk,bp,bq,pi,qi,ko,ki)
    
    #fuse axis of compute O
    bx = s[O].fuse(n,bk)
    bx = s[O].fuse(bx,bp)
    bx = s[O].fuse(bx,bq)
    
    tx = s[O].fuse(pi,qi)
    tx = s[O].fuse(tx,ko)
    
    #split outer loop
    #bx,rx = s[O].split(bx,nparts = num_sm)

    #set up the computation on shared memory
    s[shmem].set_scope("shared")
    s[shmem].compute_at(s[O],tx)
    
    #setup 128-bit aligment copy
    s[O].tensorize(kt,_int4_copy)
    #s[O].reorder(bx,tx,ki)
    #bind axis
    s[O].bind(bx,bidx)
    s[O].bind(tx,tidx)
    return(s)

def choose_best_tiling_para(output_q,output_channel):
    if(output_channel<128 or output_q<16):
       blk_q = 8
       block_row_warp = 2
       warp_col_tile = 2
       return(blk_q,block_row_warp,warp_col_tile,64)
    else:
       blk_q = 16
       block_row_warp = 4
       warp_col_tile = 4
       return(blk_q,block_row_warp,warp_col_tile,128)

