#include "stdio.h"

__global__ void whoami(void)
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.y + blockIdx.z * gridDim.x * gridDim.y;

  int blockOffset = blockId * blockDim.x * blockDim.y * blockDim.z;

  int threadOffset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  int id = blockOffset + threadOffset;

  printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
         id,
         blockIdx.x,
         blockIdx.y,
         blockIdx.z,
         blockId,
         threadIdx.x,
         threadIdx.y,
         threadIdx.z,
         threadOffset);
}

int main()
{
  const int b_x = 2, b_y = 3, b_z = 4;  // Block Dimensions
  const int t_x = 4, t_y = 4, t_z = 4;  // the max no of threads in a warp is 32, we will get 2 warps of 32 thread per block
  int blocks_Per_Grid = b_x * b_y * b_z;
  int threads_Per_Block = t_x * t_y * t_z;

  printf("%d blocks/grid\n", blocks_Per_Grid);
  printf("%d threads/block\n", threads_Per_Block);
  printf("%d total threads\n", blocks_Per_Grid * threads_Per_Block);

  dim3 blocksPerGrid(b_x, b_y, b_z);
  dim3 threadsPerBlock(t_x, t_y, t_z);

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}
