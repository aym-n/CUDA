#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

const int N = 10000000;
const int BLOCK_SIZE = 256;

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 8;
const int BLOCK_SIZE_Z = 8;

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

__global__ void addGPU3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < nx && j < ny && k < nz)
  {
    int index = i + j * nx + k * ny;
    if (index < nx * ny * nz)
      c[index] = a[index] + b[index];
  }
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
  float *h_a, *h_b, *h_CPU, *h_GPU, *h_GPU3d;
  float *d_a, *d_b, *d_c, *d_c3d;

  size_t size = N * sizeof(float);

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_CPU = (float *)malloc(size);
  h_GPU = (float *)malloc(size);
  h_GPU3d = (float *)malloc(size);

  srand(time(NULL));
  initVector(h_a, N);
  initVector(h_b, N);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  cudaMalloc(&d_c3d, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Calculates the ciel
                                                      //
  int nx = 100, ny = 100, nz = 1000;                  // N = 10000000 = 100 * 100 * 1000
  dim3 blockSize3d(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 numBlocks3d(
      (nx + blockSize3d.x - 1) / blockSize3d.x, (ny + blockSize3d.y - 1) / blockSize3d.y, (nz + blockSize3d.z - 1) / blockSize3d.z);

  printf("[Warm Up]\n");
  for (int i = 0; i < 3; i++)
  {
    addCPU(h_a, h_b, h_CPU, N);
    addGPU<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    addGPU3d<<<numBlocks3d, blockSize3d>>>(d_a, d_b, d_c3d, nx, ny, nz);
    cudaDeviceSynchronize();
  }

  printf("[Benchmarking CPU implementation]\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < 5; i++)
  {
    double start_time = get_time();
    addCPU(h_a, h_b, h_CPU, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / 5.0;

  printf("[Benchmarking GPU 1D implementation]\n");
  double gpu_1d_total_time = 0.0;
  for (int i = 0; i < 100; i++)
  {
    cudaMemset(d_c, 0, size);  // Clear previous results
    double start_time = get_time();
    addGPU<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_1d_total_time += end_time - start_time;
  }
  double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

  // Verify 1D results immediately
  cudaMemcpy(h_GPU, d_c, size, cudaMemcpyDeviceToHost);
  bool correct_1d = true;
  for (int i = 0; i < N; i++)
  {
    if (fabs(h_CPU[i] - h_GPU[i]) > 1e-4)
    {
      correct_1d = false;
      std::cout << i << " cpu: " << h_CPU[i] << " != " << h_GPU[i] << std::endl;
      break;
    }
  }
  printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

  printf("[Benchmarking GPU 3D implementation]\n");
  double gpu_3d_total_time = 0.0;
  for (int i = 0; i < 100; i++)
  {
    cudaMemset(d_c3d, 0, size);  // Clear previous results
    double start_time = get_time();
    addGPU3d<<<numBlocks3d, blockSize3d>>>(d_a, d_b, d_c3d, nx, ny, nz);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_3d_total_time += end_time - start_time;
  }
  double gpu_3d_avg_time = gpu_3d_total_time / 100.0;

  cudaMemcpy(h_GPU3d, d_c3d, size, cudaMemcpyDeviceToHost);
  bool correct_3d = true;
  for (int i = 0; i < N; i++)
  {
    if (fabs(h_CPU[i] - h_GPU3d[i]) > 1e-4)
    {
      correct_3d = false;
      std::cout << i << " cpu: " << h_CPU[i] << " != " << h_GPU3d[i] << std::endl;
      break;
    }
  }
  printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
  printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
  printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
  printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
  printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

  // Free memory
  free(h_a);
  free(h_b);
  free(h_CPU);
  free(h_GPU);
  free(h_GPU3d);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c3d);

  return 0;
}
