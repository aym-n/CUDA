#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 10000000;
const int BLOCK_SIZE = 256;

void addCPU(float *a, float *b, float *c, int n)
{
  for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

__global__ void addGPU(float *a, float *b, float *c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

void initVector(float *vec, int n)
{
  for (int i = 0; i < n; i++) vec[i] = (float)rand() / (float)RAND_MAX;
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

  size_t size = N * sizeof(float);

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_CPU = (float *)malloc(size);
  h_GPU = (float *)malloc(size);

  srand(time(NULL));
  initVector(h_a, N);
  initVector(h_b, N);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Calculates the ciel

  printf("[Warm Up]\n");
  for (int i = 0; i < 3; i++)
  {
    addCPU(h_a, h_b, h_CPU, N);
    addGPU<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  printf("[Benchmarking CPU implementation]\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++)
  {
    double start_time = get_time();
    addCPU(h_a, h_b, h_CPU, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 20.0;

  printf("[Benchmarking GPU implementation]\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++)
  {
    double start_time = get_time();
    addGPU<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / 20.0;

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  cudaMemcpy(h_GPU, d_c, size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < N; i++)
  {
    if (fabs(h_GPU[i] - h_CPU[i]) > 1e-5)
    {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  free(h_a);
  free(h_b);
  free(h_CPU);
  free(h_GPU);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
