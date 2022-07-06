#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "cpu_bitmap.h"

# define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr)
{
	// 将idx映射到像素位置
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// 使用共享内存缓冲区来保存计算结果
	// 16*16的线程块中的每个线程在该缓冲区都有一个对应位置
	__shared__ float shared[16][16];

	// 计算该位置的值
	const float period = 128.0f;
	// 写入shared[][]
	shared[threadIdx.x][threadIdx.y] =
		255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
		(sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

	// 一个重要的同步点
	// 如果不在这里同步，线程将shared[][]的结果保存到像素中时，负责写入shared[][]的线程可能还没有写完。
	__syncthreads();

	// 保存到像素
	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}


int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	kernel << <grids, threads >> > (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	HANDLE_ERROR(cudaFree(dev_bitmap));
}