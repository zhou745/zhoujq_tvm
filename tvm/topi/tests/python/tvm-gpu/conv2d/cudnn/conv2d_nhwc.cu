#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <cfloat>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "device_launch_parameters.h"

/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    cudaDeviceReset();                                                \
    exit(1);                                                          \
  } while (0)

#define checkCUDNN(status)                                        \
  do {                                                            \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                   \
    }                                                             \
  } while (0)

#define checkCudaErrors(status)             \
  do {                                      \
    std::stringstream _error;               \
    if (status != 0) {                      \
      _error << "Cuda failure: " << status; \
      FatalError(_error.str());             \
    }                                       \
  } while (0)

/** Convolutional layer */
struct ConvolutionLayer {
  int kernel_size;
  int in_channels, in_height, in_width;
  int out_channels, out_height, out_width;
  std::vector<int8_t> pconv;

  int pad_height;
  int pad_width;
  int stride_h;
  int stride_v;
  int dilation_h;
  int dilation_w;

  ConvolutionLayer(int in_channels_, int out_channels_, int kernel_size_,
                   int in_w_, int in_h_, int pad, int stride, int dilation)
      : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;

    pad_height = pad_width = pad;
    stride_h = stride_v = stride;
    dilation_h = dilation_w = dilation;
  }
};

/** Training context */
struct TrainingContext {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;
  cudnnFilterDescriptor_t conv1filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc;
  cudnnConvolutionFwdAlgo_t conv1algo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;

  cudaEvent_t start, stop;

  double sum = 0.0;

  // Disable copying
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;

  // Constructor
  TrainingContext(int gpuid, int batch_size, ConvolutionLayer& conv1)
      : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    /** Create descriptors within the constructor.
     * As instructed in the Usual manual, descriptors for
     * input and output tensors, filter, and the forward
     * convolution operator are created along with
     * cuDNN handle.
     */
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));

    // Initialize convolution forward pass
    size_t workspaceSizeFromConv = SetFwdConvolutionTensors(
        conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo);
    m_workspaceSize = std::max((int)workspaceSizeFromConv, 0);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~TrainingContext() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
  }

  /** Set tensors and ops for forward pass */
  size_t SetFwdConvolutionTensors(ConvolutionLayer& conv,
                                  cudnnTensorDescriptor_t& srcTensorDesc,
                                  cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc,
                                  cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionFwdAlgo_t& algo) {
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NHWC,
                                          CUDNN_DATA_INT8, n, c, h, w));

    checkCUDNN(cudnnSetFilter4dDescriptor(
        filterDesc, CUDNN_DATA_INT8, CUDNN_TENSOR_NHWC, conv.out_channels,
        conv.in_channels, conv.kernel_size, conv.kernel_size));

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convDesc, conv.pad_height, conv.pad_width, conv.stride_h, conv.stride_v,
        conv.dilation_h, conv.dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_INT32));

    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NHWC,
                                          CUDNN_DATA_INT8, n, c, h, w));

    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo,
        &sizeInBytes));

    return sizeInBytes;
  }

  /** Execute forward pass */
  void ForwardPropagation(void* data, void* result, void* weights,
                          void* workspace) {
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(cudaSetDevice(m_gpuid));

    cudaEventRecord(start, 0);
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor, data,
                                       conv1filterDesc, weights, conv1Desc,
                                       conv1algo, workspace, m_workspaceSize,
                                       &beta, conv1Tensor, result));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0f;
    sum += elapsed;
  }
};

struct Workload {
  int iterations;
  int batch_size;
  int width, height, channels;
  int out_channels, kernel_size;
  int pad, stride, dilation;
};

void test(const Workload& wkl) {
  auto gpu = 0;
  auto channels = wkl.channels;
  auto out_channels = wkl.out_channels;
  auto kernel_size = wkl.kernel_size;
  auto width = wkl.width;
  auto height = wkl.height;
  auto pad = wkl.pad;
  auto stride = wkl.stride;
  auto dilation = wkl.dilation;
  auto batch_size = wkl.batch_size;
  auto iterations = wkl.iterations;

  ConvolutionLayer conv1(channels, out_channels, kernel_size, width, height,
                         pad, stride, dilation);
  TrainingContext context(gpu, batch_size, conv1);

  // Initizlie convolution weight
  std::mt19937 g(42);
  std::uniform_int_distribution<int8_t> dconv1(-128, 127);

  for (auto& iter : conv1.pconv) {
    iter = dconv1(g);
  }

  // Initailize input image (batch size = 1)
  std::vector<int8_t> img_float(1 * width * height * channels);
  for (auto& iter : img_float) {
    iter = dconv1(g);
  }

  // Allocate input and output on GPU; copy input over to GPU
  int8_t *d_data, *d_conv1;
  checkCudaErrors(cudaMalloc(&d_data, sizeof(int8_t) * context.m_batchSize *
                                          channels * height * width));
  checkCudaErrors(cudaMalloc(&d_conv1, sizeof(int32_t) * context.m_batchSize *
                                           conv1.out_channels *
                                           conv1.out_height * conv1.out_width));
  checkCudaErrors(cudaMemcpyAsync(
      d_data, &img_float[0], sizeof(int8_t) * 1 * channels * width * height,
      cudaMemcpyHostToDevice));

  // Allocate kernel on GPU
  float* d_pconv1;
  checkCudaErrors(cudaMalloc(&d_pconv1, sizeof(int8_t) * conv1.pconv.size()));
  checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],
                                  sizeof(int8_t) * conv1.pconv.size(),
                                  cudaMemcpyHostToDevice));

  // Temporary buffers and workspaces
  void* d_cudnn_workspace = nullptr;
  if (context.m_workspaceSize > 0) {
    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));
  }

  // Start forward pass
  checkCudaErrors(cudaDeviceSynchronize());
  for (int iter = 0; iter < iterations; ++iter) {
    context.ForwardPropagation(d_data, d_conv1, d_pconv1, d_cudnn_workspace);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  auto sum = context.sum;

  auto num_flops = (long long)width * height * channels * out_channels *
                   batch_size * 2 * kernel_size * kernel_size;
  auto GFLOPS = num_flops * 1.0 / sum * iterations / 1e9;
  printf("%d %dx%d %d %d kernel %d Time: %f s, Time/Iter %f s, %.2f GFLOPS\n",
         batch_size, width, height, channels, out_channels, kernel_size, sum,
         sum / iterations, GFLOPS);

  // Free data structures
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_conv1));
  checkCudaErrors(cudaFree(d_pconv1));

  if (d_cudnn_workspace != nullptr)
    checkCudaErrors(cudaFree(d_cudnn_workspace));
}

int main() {
  // iterations, N, H, W, C_in, C_out, kernel, padding, stride, dilation
  Workload wkls[]{{1000, 1, 7, 7, 512, 512, 1, 0, 1, 1},
                  {1000, 4, 7, 7, 512, 512, 1, 0, 1, 1},
                  {100, 128, 7, 7, 512, 512, 1, 0, 1, 1},
                  {1000, 1, 7, 7, 512, 512, 3, 1, 1, 1},
                  {1000, 4, 7, 7, 512, 512, 3, 1, 1, 1},
                  {100, 128, 7, 7, 512, 512, 3, 1, 1, 1},

                  {1000, 1, 14, 14, 256, 256, 1, 0, 1, 1},
                  {1000, 4, 14, 14, 256, 256, 1, 0, 1, 1},
                  {100, 128, 14, 14, 256, 256, 1, 0, 1, 1},
                  {1000, 1, 14, 14, 256, 256, 3, 1, 1, 1},
                  {1000, 4, 14, 14, 256, 256, 3, 1, 1, 1},
                  {100, 128, 14, 14, 256, 256, 3, 1, 1, 1},

                  {1000, 1, 56, 56, 64, 64, 1, 0, 1, 1},
                  {1000, 4, 56, 56, 64, 64, 1, 0, 1, 1},
                  {10, 128, 56, 56, 64, 64, 1, 0, 1, 1},
                  {1000, 1, 56, 56, 64, 64, 3, 1, 1, 1},
                  {1000, 4, 56, 56, 64, 64, 3, 1, 1, 1},
                  {10, 128, 56, 56, 64, 64, 3, 1, 1, 1}};

  for (auto&& wkl : wkls) test(wkl);
  return 0;
}
