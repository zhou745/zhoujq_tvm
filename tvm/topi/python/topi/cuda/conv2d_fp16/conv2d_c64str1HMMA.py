import tvm
from ..tensor_intrin import int4_copy
#name the thread and block axis
bidx=tvm.thread_axis('blockIdx.x')   
tidx=tvm.thread_axis('threadIdx.x')

def conv2d_c64str1HMMA(cfg, data, kernel,data_shape,kernel_shape,output_shape,dilation,dir_path):
    #print the size of current kernel
    print("data size %s:"%data.name,data_shape)
    print("kernel size in layer %s:"%kernel.name,kernel_shape)
    print(output_shape)
    fortype="unroll"

    #block_para
    blk_q = 16
    blk_p = 16
    blk_size_r=64
    blk_size_c = blk_p*blk_q
    block_row_warp = 1
    warp_col_tile = 4
    warp_row_tile = 4
    block_col_warp = 4

    #offset preset
    shieft =8
    offset_D_im2col = (2+blk_q)*(2+blk_p)*16
    offset_F = offset_D_im2col+(shieft+16)*blk_size_c

    #shared memory usage
    output_copy = blk_size_r*blk_size_c
    im2col_use = offset_D_im2col+(shieft+16)*blk_size_c+(shieft+16)*blk_size_r
    shmem_use = max(output_copy,im2col_use)

    def convolutionfp16(D,F,shmem):
        #ir builder for constructing the main body
        ib = tvm.ir_builder.create()
        
        #id of current warp and offset of shared memory when storing
        warpid=tidx/32
        warp_offset_output = warpid%block_row_warp*16*warp_row_tile\
        +warpid/block_row_warp*warp_col_tile*block_row_warp*warp_row_tile*256

        #include files 
        include_file=tvm.call_intrin("float32","include_cpp_head",dir_path+"/conv2d_HMMA.h")
        ib.emit(include_file)

        #declare the matrix fragment
        declare_a = tvm.call_intrin("float32","wmma_fragment","matrix_a","half","row_major","a_frag",warp_col_tile)
        declare_b = tvm.call_intrin("float32","wmma_fragment","matrix_b","half","col_major","b_frag",warp_row_tile)
        declare_c = tvm.call_intrin("float32","wmma_fragment","accumulator","half","c_frag",warp_col_tile,warp_row_tile)
        ib.emit(declare_a)
        ib.emit(declare_b)
        ib.emit(declare_c)

        #define the shared memory for loading data and offset for loading the data
        offset_D_warp = offset_D_im2col+tidx*(16+shieft)*2
        offset_F_warp = offset_F+tidx/2*(16+shieft)+tidx%2*8

        #ir template for thread synchronization
        sync = tvm.call_extern("float32","__syncthreads")

        #main for conducting the computation
        #set the pointer to first address of D
        Dp=D.access_ptr("r")
        Sp=shmem.access_ptr("r")
        Fp=F.access_ptr("r")

        #load the first data from global memory for the reuse of 9 times
        load_first_data = tvm.call_extern("float32","load_matrix_D",Dp,Sp,\
                                            output_shape[0],data_shape[1],data_shape[2],data_shape[3],0,dilation,1)
        ib.emit(load_first_data)

        #load the first filter from global memory:
        load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,kernel_shape[0],\
                                    kernel_shape[3],data_shape[0],data_shape[1],data_shape[2],tidx%2*8,0,0)
        ib.emit(load_filter)
        #fill fragment c with 0
        with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
            with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:              
                fill_c_zero = tvm.call_intrin("float","wmma_fill_fragment","c_frag",col_id,row_id,"half",0.)
                ib.emit(fill_c_zero)
        ib.emit(sync)

        with ib.for_range(0,data_shape[3]/16,name="c_id",for_type=fortype) as c_id:
            with ib.for_range(0,9,name="ker_id",for_type=fortype) as ker_id:
                #now load matrix fragment
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    load_matrix_frag_F = tvm.call_intrin("float32","wmma_load_matrix_sync","a_frag",col,Sp,\
                                                    tidx/(32*block_row_warp)*\
                                                    (16*warp_col_tile*(blk_q+2))+(col+ker_id/3)*(16*(blk_q+2))\
                                                    +ker_id%3*16,16)
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
                    load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,kernel_shape[0],kernel_shape[3],\
                                                data_shape[0],data_shape[1],data_shape[2],c_id*16+tidx%2*8,ker_id+1,0)
                    ib.emit(load_filter)
                    ib.emit(sync)

            
            with ib.if_scope(c_id<data_shape[3]/16-1):
                #load the next 9 iteration data from global memory
                load_data = tvm.call_extern("float32","load_matrix_D",Dp,Sp,\
                                output_shape[0],output_shape[1],output_shape[2],data_shape[3],c_id*16+16,dilation,1)
                ib.emit(load_data)

                #load filter for next cd iter
                load_filter=tvm.call_extern("float32","load_matrix_F",Fp,Sp,offset_F_warp,kernel_shape[0],\
                                            data_shape[3],data_shape[0],data_shape[1],data_shape[2],c_id*16+16+tidx%2*8,0,0)
                ib.emit(load_filter)
                ib.emit(sync)
        with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
            with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                store_c_fragment = tvm.call_intrin("float32","wmma_store_matrix_sync",Sp,warp_offset_output+col_id*(256*warp_row_tile*block_row_warp)+row_id*16,"c_frag",col_id,row_id,blk_size_r)
                ib.emit(store_c_fragment)
        ib.emit(sync)
        body = ib.get()
        return(body)

    shmem = tvm.extern((shmem_use,),[data,kernel],lambda ins,outs:convolutionfp16(ins[0],ins[1],outs[0]),\
                        name = "shmem",dtype = 'float16',\
                        out_buffers=tvm.decl_buffer((shmem_use,),dtype='float16',scope='shared',offset_factor=1))

    O = tvm.compute(output_shape,lambda dn,dp,dq,dk:shmem[dk%blk_size_r+dp/dilation%blk_p*blk_q*blk_size_r+dq/dilation%blk_q*blk_size_r],tag="conv2d_NHWC_HMMA",\
                    attrs={"blk_size":blk_size_r,"dilation":dilation,"version":1})

    num_flop =  data_shape[0]* output_shape[2]*output_shape[3]*kernel_shape[0]*2*data_shape[3]*kernel_shape[1]*kernel_shape[2]
    cfg.add_flop(num_flop)
    
    return(O)


def schedule_conv2d_fp16_c64str1HMMA(cfg,s,O):
    blk_p=16
    blk_q=16
    ki_factor=8

    blk_size_r=O.op.attrs["blk_size"]
    dilation_size=O.op.attrs["dilation"]

    print("blk_size=%d"%(blk_size_r))
    print("dilation size is %d"%dilation_size)

    shmem = O.op.input_tensors[0]
    n,p,q,k = s[O].op.axis
    
    #split the axis for computation
    bp,pi = s[O].split(p,factor = blk_p*dilation_size)
    bq,qi = s[O].split(q,factor = blk_q*dilation_size)
    pi,dp = s[O].split(pi,factor = dilation_size)
    qi,dq = s[O].split(qi,factor = dilation_size)
    bk,ki = s[O].split(k,factor = blk_size_r)
    ki,kt = s[O].split(ki,factor = ki_factor)

    s[O].reorder(n,bk,bp,bq,dp,dq,pi,qi,ki,kt)

    #fuse axis of compute O
    bx = s[O].fuse(n,bk)
    bx = s[O].fuse(bx,bp)
    bx = s[O].fuse(bx,bq)
    bx = s[O].fuse(bx,dp)
    bx = s[O].fuse(bx,dq)
    
    tx = s[O].fuse(pi,qi)
    tx = s[O].fuse(tx,ki)  
    tx,ko = s[O].split(tx,nparts = 128)

    s[shmem].set_scope("shared")
    s[shmem].compute_at(s[O],tx)
    
    #setup 128-bit aligment copy

    s[O].vectorize(kt)
    #s[O].reorder(bx,tx,ki)
    #bind axis
    s[O].bind(bx,bidx)
    s[O].bind(tx,tidx)
    
    return s