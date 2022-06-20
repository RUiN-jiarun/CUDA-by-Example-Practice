#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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

int main()
{
	test();
	return 0;
}