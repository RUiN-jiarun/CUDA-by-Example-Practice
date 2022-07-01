#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N (15 * 1024)		// FIXME: 33*1024会越界

// 硬件将线程块的数量限制为不超过65535
// 启动和函数时每个线程块的线程数量也不应该超过设备属性中的maxThreadsPerBlock，这一般是512

__global__ void add(int* a, int* b, int* c)
{
	//int tid = blockIdx.x;	// 计算该索引处的数据
	//int tid = threadIdx.x;	// 只有一个线程块，通过县城索引对数据进行索引
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// 线程块block与线程thread的二维组织形式
	//         ----------------------------------
	// block 0 | thread 0 | thread 1 | thread 2 |
	//         ----------------------------------
	// block 1 | thread 0 | thread 1 | thread 2 |
	//         ----------------------------------
	// block 2 | thread 0 | thread 1 | thread 2 |
	//         ----------------------------------
	
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;		// 每次地新增数量为GPU并行线程数
	}
}


int main()
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// 在GPU上分配内存
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// 在CPU上为数组a[] b[]赋值
	// 调整了一些数值，防止越界
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	// a b复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// 在主机代码执行add()中的设备代码
	// 分配了并行线程块的数量为N
	//add << <N, 1 >> > (dev_a, dev_b, dev_c);
	add << <128, 128 >> > (dev_a, dev_b, dev_c);	// 分配并行线程块的数量为128，每一个线程块中有128个线程

	// c复制到CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if ((a[i] + b[i]) != c[i])
		{
			printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success)    printf("All correct.\n");

	// 释放GPU内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}