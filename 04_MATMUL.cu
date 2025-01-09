#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int M = 256;
const int K = 256;
const int N = 256;

const int BLOCK_SIZE = 32;

void matmulCPU(float *A, float *B, float *C, int m, int n, int k)
{
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
      C[i * n + j] = sum;
    }
  }
}

__global__ void matmulGPU(float *A, float *B, float *C, int m, int n, int k)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n)
  {
    float sum = 0.0f;
    for (int l = 0; l < k; l++) sum += A[row * k + l] * B[l * n + col];
    C[row * n + col] = sum;
  }
}

void initMatrix(float *vec, int rows, int cols)
{
  for (int i = 0; i < rows * cols; i++) vec[i] = (float)rand() / (float)RAND_MAX;
}

double get_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
  float *h_a, *h_b, *h_CPU, *h_GPU;
  float *d_a, *d_b, *d_c;

  int size_A = M * K * sizeof(float);
  int size_B = K * N * sizeof(float);
  int size_C = M * N * sizeof(float);

  h_a = (float *)malloc(size_A);
  h_b = (float *)malloc(size_B);
  h_CPU = (float *)malloc(size_C);
  h_GPU = (float *)malloc(size_C);

  srand(time(NULL));
  initMatrix(h_a, M, K);
  initMatrix(h_b, K, N);

  cudaMalloc(&d_a, size_A);
  cudaMalloc(&d_b, size_B);
  cudaMalloc(&d_c, size_C);

  cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  printf("[Warm Up]\n");
  for (int i = 0; i < 3; i++)
  {
    matmulCPU(h_a, h_b, h_CPU, M, K, N);
    matmulGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
  }

  printf("[Benchmarking CPU implementation]\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++)
  {
    double start_time = get_time();
    matmulCPU(h_a, h_b, h_CPU, M, K, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  printf("[Benchmarking GPU implementation]\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++)
  {
    double start_time = get_time();
    matmulGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
  printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  free(h_a);
  free(h_b);
  free(h_CPU);
  free(h_GPU);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
