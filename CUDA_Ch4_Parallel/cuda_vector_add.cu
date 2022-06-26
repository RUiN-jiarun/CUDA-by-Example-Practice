#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N 10

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x;	// 计算该索引处的数据
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
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
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	
	// a b复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// 在主机代码执行add()中的设备代码
	// 分配了并行线程块的数量为N
	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// c复制到CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// 释放GPU内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}