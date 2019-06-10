#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void fuse_conv2d_1_kernel0( half* __restrict__ input1,  half* __restrict__ input0,  half* __restrict__ O) {
   half conv[128];
  __shared__ half input1_shared[7168];
  __shared__ half data_pad_shared[14896];
  for (int dp_init = 0; dp_init < 2; ++dp_init) {
    for (int dq_init = 0; dq_init < 2; ++dq_init) {
      for (int dk_init = 0; dk_init < 4; ++dk_init) {
        conv[(((dp_init * 8) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((64 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((16 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((80 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((32 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((96 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((48 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
        conv[(((112 + (dp_init * 8)) + (dq_init * 4)) + dk_init)] = __float2half_rn(0.000000e+00f);
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer) {
    for (int ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
      if ((((int)threadIdx.y) * 2) < ((224 - ((int)threadIdx.x)) - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer * 64))) {
        if (((int)threadIdx.y) < (112 - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer * 32))) {
          if ((((int)blockIdx.x) * 32) < (64 - ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer * 64)) / 7))) {
            input1_shared[((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 16)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer * 1024)) + ax3_inner)] = input1[((((((int)blockIdx.x) * 25088) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer * 64)) / 7) * 784)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer) % 7) * 112)) + ax3_inner)];
          }
        }
      }
    }
  }
  for (int rw_outer_outer = 0; rw_outer_outer < 6; ++rw_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 < 15; ++ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1) {
      for (int ax3_inner1 = 0; ax3_inner1 < 16; ++ax3_inner1) {
        if ((((int)threadIdx.y) * 2) < ((931 - ((int)threadIdx.x)) - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64))) {
          if (((int)threadIdx.y) < (466 - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 32))) {
            data_pad_shared[((((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 931) * 14896) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1) % 7) * 16)) + ((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 7) % 133) * 112)) + ax3_inner1)] = ((((((3 - (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 7) % 133)) <= ((((int)blockIdx.y) / 32) * 128)) && (((((int)blockIdx.y) / 32) * 128) < (259 - (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 7) % 133)))) && (((3 - rw_outer_outer) - ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1) % 7)) <= ((((int)blockIdx.y) % 32) * 8))) && (((((int)blockIdx.y) % 32) * 8) < ((259 - rw_outer_outer) - ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1) % 7)))) ? input0[(((((((((((int)blockIdx.y) / 32) * 524288) + ((((int)blockIdx.y) % 32) * 128)) + (rw_outer_outer * 16)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 931) * 1048576)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1) % 7) * 16)) + ((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer1 * 64)) / 7) % 133) * 4096)) + ax3_inner1) - 12336)] : __float2half_rn(0.000000e+00f));
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 = 0; ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 < 4; ++ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2) {
      for (int ax3_inner2 = 0; ax3_inner2 < 16; ++ax3_inner2) {
        if ((((int)threadIdx.y) * 2) < ((224 - ((int)threadIdx.x)) - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 * 64))) {
          if (((int)threadIdx.y) < (112 - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 * 32))) {
            if ((((int)blockIdx.x) * 32) < (64 - ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 * 64)) / 7))) {
              input1_shared[(((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 16)) + (((1 + rw_outer_outer) % 2) * 3584)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 * 1024)) + ax3_inner2)] = input1[(((((16 + (((int)blockIdx.x) * 25088)) + (rw_outer_outer * 16)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2 * 64)) / 7) * 784)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer2) % 7) * 112)) + ax3_inner2)];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rh_inner = 0; rh_inner < 7; ++rh_inner) {
      for (int rc_inner_inner = 0; rc_inner_inner < 16; ++rc_inner_inner) {
        for (int dp = 0; dp < 2; ++dp) {
          for (int dq = 0; dq < 2; ++dq) {
            for (int dk = 0; dk < 4; ++dk) {
              conv[(((dp * 8) + (dq * 4)) + dk)] = (conv[(((dp * 8) + (dq * 4)) + dk)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((((int)threadIdx.x) * 448) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((64 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((64 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((((int)threadIdx.x) * 448) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((16 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((16 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((896 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((80 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((80 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((896 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((32 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((32 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((1792 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((96 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((96 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((1792 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((48 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((48 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((2688 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
              conv[(((112 + (dp * 8)) + (dq * 4)) + dk)] = (conv[(((112 + (dp * 8)) + (dq * 4)) + dk)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner * 112)) + rc_inner_inner) + (dp * 224)) + (dq * 32))] * input1_shared[(((((2688 + (((int)threadIdx.x) * 448)) + ((rw_outer_outer % 2) * 3584)) + (rh_inner * 16)) + rc_inner_inner) + (dk * 112))]));
            }
          }
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 = 0; ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 < 15; ++ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3) {
    for (int ax3_inner3 = 0; ax3_inner3 < 16; ++ax3_inner3) {
      if ((((int)threadIdx.y) * 2) < ((931 - ((int)threadIdx.x)) - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64))) {
        if (((int)threadIdx.y) < (466 - (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 32))) {
          data_pad_shared[((((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 931) * 14896) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3) % 7) * 16)) + ((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 7) % 133) * 112)) + ax3_inner3)] = (((((3 - (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 7) % 133)) <= ((((int)blockIdx.y) / 32) * 128)) && (((((int)blockIdx.y) / 32) * 128) < (259 - (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 7) % 133)))) && (((((int)blockIdx.y) % 32) * 8) < (253 - ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3) % 7)))) ? input0[((((((((((int)blockIdx.y) / 32) * 524288) + ((((int)blockIdx.y) % 32) * 128)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 931) * 1048576)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3) % 7) * 16)) + ((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) + (ax0_ax1_fused_ax2_fused_ax3_outer_fused_outer_outer_outer3 * 64)) / 7) % 133) * 4096)) + ax3_inner3) - 12240)] : __float2half_rn(0.000000e+00f));
        }
      }
    }
  }
  __syncthreads();
  for (int rh_inner1 = 0; rh_inner1 < 7; ++rh_inner1) {
    for (int rc_inner_inner1 = 0; rc_inner_inner1 < 16; ++rc_inner_inner1) {
      for (int dp1 = 0; dp1 < 2; ++dp1) {
        for (int dq1 = 0; dq1 < 2; ++dq1) {
          for (int dk1 = 0; dk1 < 4; ++dk1) {
            conv[(((dp1 * 8) + (dq1 * 4)) + dk1)] = (conv[(((dp1 * 8) + (dq1 * 4)) + dk1)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((((int)threadIdx.x) * 448) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((64 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((64 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((((int)threadIdx.x) * 448) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((16 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((16 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((896 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((80 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((80 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((896 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((32 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((32 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((1792 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((96 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((96 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((1792 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((48 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((48 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[(((((((((int)threadIdx.y) / 2) * 448) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((2688 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
            conv[(((112 + (dp1 * 8)) + (dq1 * 4)) + dk1)] = (conv[(((112 + (dp1 * 8)) + (dq1 * 4)) + dk1)] + (data_pad_shared[((((((7168 + ((((int)threadIdx.y) / 2) * 448)) + ((((int)threadIdx.y) % 2) * 64)) + (rh_inner1 * 112)) + rc_inner_inner1) + (dp1 * 224)) + (dq1 * 32))] * input1_shared[((((2688 + (((int)threadIdx.x) * 448)) + (rh_inner1 * 16)) + rc_inner_inner1) + (dk1 * 112))]));
          }
        }
      }
    }
  }
  for (int dp_inner_inner_inner = 0; dp_inner_inner_inner < 2; ++dp_inner_inner_inner) {
    for (int dq_inner_inner_inner = 0; dq_inner_inner_inner < 2; ++dq_inner_inner_inner) {
      (( int2*)(O + (((((((((((int)blockIdx.y) / 32) * 524288) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((dp_inner_inner_inner * 8) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((262144 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((64 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((8 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((16 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((262152 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((80 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((16 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((32 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((262160 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((96 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((24 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((48 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
      (( int2*)(O + ((((((((262168 + ((((int)blockIdx.y) / 32) * 524288)) + ((((int)blockIdx.y) % 32) * 256)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.y) / 2) * 16384)) + ((((int)threadIdx.y) % 2) * 128)) + (((int)threadIdx.x) * 4)) + (dp_inner_inner_inner * 8192)) + (dq_inner_inner_inner * 64))))[0] = (( int2*)(conv + ((112 + (dp_inner_inner_inner * 8)) + (dq_inner_inner_inner * 4))))[0];
    }
  }
}

