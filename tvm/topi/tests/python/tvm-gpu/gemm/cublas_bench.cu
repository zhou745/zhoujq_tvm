/*
Compile: nvcc cublas_bench.cu -std=c++11 -arch=sm_61 -lcublas -o cublas_bench
Usage: ./cublas_bench N
*/

#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

using namespace std;

typedef signed char int8;
typedef int int32;

const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
template <typename T>
void CPU_fill_rand(T *A, int nr_rows_A, int nr_cols_A) {
  int a = 1;

  for (int i = 0; i < nr_rows_A * nr_cols_A; i++) {
    A[i] = static_cast<T>(rand() / (float)(RAND_MAX / a));
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << argv[0] << " N" << std::endl;
    return 1;
  }

  int N = atoi(argv[1]);
  int repeats = 1000;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  int8 *h_A = (int8 *)malloc(N * N * sizeof(int8));
  int8 *h_B = (int8 *)malloc(N * N * sizeof(int8));
  int32 *h_C = (int32 *)malloc(N * N * sizeof(int32));

  int8 *d_A, *d_B;
  int32 *d_C;

  CPU_fill_rand(h_A, N, N);
  CPU_fill_rand(h_B, N, N);
  CPU_fill_rand(h_C, N, N);

  checkCuda(cudaMalloc(&d_A, N * N * sizeof(int8)));
  checkCuda(cudaMalloc(&d_B, N * N * sizeof(int8)));
  checkCuda(cudaMalloc(&d_C, N * N * sizeof(int32)));

  checkCuda(cudaMemcpy(d_A, h_A, N * N * sizeof(int8), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_B, h_B, N * N * sizeof(int8), cudaMemcpyHostToDevice));
  checkCuda(
      cudaMemcpy(d_C, h_C, N * N * sizeof(int32), cudaMemcpyHostToDevice));

  int lda, ldb, ldc, m, n, k;
  const int alf = 1;
  const int bet = 0;
  const auto *alpha = &alf;
  const auto *beta = &bet;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto sum = 0.0;
  for (int rep = 0; rep < repeats; rep++) {
    cudaEventRecord(start, 0);

    m = n = k = lda = ldb = ldc = N;
    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A,
                        CUDA_R_8I, lda, d_B, CUDA_R_8I, ldb, beta, d_C,
                        CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if (stat != CUBLAS_STATUS_SUCCESS) {
      cout << cublasGetErrorString(stat) << endl;
      exit(-1);
    }

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0f;
    sum += elapsed;
  }
  long long num_flops = (long long)N * N * N * 2;
  auto GFLOPS = num_flops / (sum / repeats) / 1e9;

  cout << "int8: size " << N << " average: " << sum / repeats << " s " << GFLOPS
       << " GFLOPS" << endl;

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
  cublasDestroy(handle);
  return 0;
}
