#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

__global__ void hello_world_from_gpu(void)
{
	printf("Hello World from GPU\n");
	return;
}

void test()
{
	printf("Hello World from CPU\n");
	hello_world_from_gpu << < 1, 5 >> > ();
	cudaDeviceReset();
}

// 设备代码
// 可以在设备代码中适用cudaMalloc()分配的指针进行内存读写，但不可以在主机代码中这样做
__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

/// <summary>
/// 将参数传递给kernel函数，主机代码
/// </summary>
void test_add()
{
	int c;
	int* dev_c;
	// 在CUDA运行时在设备上分配内存
	// cudaMalloc(ptr_to_var, sizeof_memory);
	// HANDLE_ERROR是book.h中的一个宏，用于判断是否返回了错误值
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

	
	add << < 1, 1 >> > (2, 7, dev_c);

	// 主机代码调用cudaMemcpy()来方位设备上的内存
	// 参数cudaMemcpyDeviceToHost说明源指针是设备指针，目标指针是主机指针
	// 类似地还有cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice。但是若两个指针都在设备上，直接适用memcpy()函数
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);

	// 释放内存
	cudaFree(dev_c);
}


int main()
{
	test();
	test_add();
	return 0;
}