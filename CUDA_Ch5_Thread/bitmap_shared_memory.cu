#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "cpu_bitmap.h"

# define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr)
{
	// ��idxӳ�䵽����λ��
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// ʹ�ù����ڴ滺���������������
	// 16*16���߳̿��е�ÿ���߳��ڸû���������һ����Ӧλ��
	__shared__ float shared[16][16];

	// �����λ�õ�ֵ
	const float period = 128.0f;
	// д��shared[][]
	shared[threadIdx.x][threadIdx.y] =
		255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
		(sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

	// һ����Ҫ��ͬ����
	// �����������ͬ�����߳̽�shared[][]�Ľ�����浽������ʱ������д��shared[][]���߳̿��ܻ�û��д�ꡣ
	__syncthreads();

	// ���浽����
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