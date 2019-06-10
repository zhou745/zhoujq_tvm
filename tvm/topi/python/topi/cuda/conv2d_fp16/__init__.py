import os 
from ...util import get_const_tuple
from .conv2d_c16str1HMMA import conv2d_c16str1HMMA,schedule_conv2d_fp16_c16str1HMMA
from .conv2d_c64str1HMMA import conv2d_c64str1HMMA,schedule_conv2d_fp16_c64str1HMMA
from .conv2d_str2HFMA import conv2d_NHWC_HFMA,schedule_conv2d_NHWC_HFMA
from .conv2d_Tensorcore_copy import conv2d_Tensor_core_copy,schedule_conv2d_Tensor_core_copy
dir_path = os.path.dirname(os.path.realpath(__file__))

def conv2d_NHWC_fp16(cfg, data, kernel, stride, padding, dilation,layout,out_dtype):
    """Convolution operator in NHWC layout for fp16.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template, used when stride >=2

    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    kernel : tvm.Tensor
        4-D with shape [num_filter, filter_height, filter_width, in_channel]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel_block, out_channel_chunk]
    """
    #check layout
    assert layout=="NHWC"

    #get stride
    if isinstance(stride, int):
        stride_l = [stride,stride]
    else:
        stride_l = stride
    #get padding
    if isinstance(padding,int):
        padding_l = [padding,padding]
    else:
        padding_l = padding
    #get dilation
    if isinstance(dilation,int):
        dilation_l=dilation
    else:
        dilation_l=dilation[0]
        print("dilation only supports for integer, the first number in the list is choosing")
    
    #get data and kernel shape
    data_b, data_h, data_w,data_c = get_const_tuple(data.shape)
    kernel_n,kernel_r, kernel_s,kernel_c = get_const_tuple(kernel.shape)

    #compute the out pust
    out_h = (data_h - (kernel_r - 1) * dilation_l - 1 + padding_l[0]*2) // stride_l[0] + 1
    out_w = (data_w - (kernel_s - 1) * dilation_l - 1 + padding_l[1]*2) // stride_l[1] + 1
    
    data_shape = (data_b, data_h, data_w,data_c)
    kernel_shape = (kernel_n,kernel_r, kernel_s,kernel_c)
    output_shape = (data_b, out_h, out_w, kernel_n)

    #choose different kernel by the output and in channel and stride
    if dilation_l ==0:
        output = conv2d_Tensor_core_copy(cfg,data,data_shape)
    elif stride_l[0] ==1 and stride_l[1]==1 and data_c<=128 and kernel_r==3 and kernel_s==3:
        # kernel for small channel size
        if data_c<=128:
            output=conv2d_c16str1HMMA(cfg, data, kernel,data_shape,kernel_shape,\
                                        output_shape,dilation_l,dir_path)
        else:
            output=conv2d_c64str1HMMA(cfg, data, kernel, data_shape, kernel_shape,\
                                       output_shape,dilation_l,dir_path)
    else:
            output=conv2d_NHWC_HFMA(cfg,data,kernel,data_shape,kernel_shape,output_shape,stride_l,padding_l) 
        
    return(output)

def schedule_NHWC_fp16(cfg,s,O):
    
    if int(O.op.attrs["version"])==0:
        print("using schedule version 0")
        s = schedule_conv2d_fp16_c16str1HMMA(cfg,s,O)
    elif int(O.op.attrs["version"])==1:
        print("using schedule version 1")
        s = schedule_conv2d_fp16_c64str1HMMA(cfg,s,O)
    elif int(O.op.attrs["version"])==2:
        print("using schedule version 2")
        s = schedule_conv2d_NHWC_HFMA(cfg,s,O)
    else:
        print("using schedule version 3")
        s = schedule_conv2d_Tensor_core_copy(cfg,s,O)
    return(s)