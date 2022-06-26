#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N 10

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x;	// �����������������
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}


int main()
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// ��GPU�Ϸ����ڴ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// ��CPU��Ϊ����a[] b[]��ֵ
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	
	// a b���Ƶ�GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// ����������ִ��add()�е��豸����
	// �����˲����߳̿������ΪN
	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// c���Ƶ�CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// �ͷ�GPU�ڴ�
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}