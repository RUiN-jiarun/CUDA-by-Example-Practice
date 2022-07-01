#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N (15 * 1024)		// FIXME: 33*1024��Խ��

// Ӳ�����߳̿����������Ϊ������65535
// �����ͺ���ʱÿ���߳̿���߳�����Ҳ��Ӧ�ó����豸�����е�maxThreadsPerBlock����һ����512

__global__ void add(int* a, int* b, int* c)
{
	//int tid = blockIdx.x;	// �����������������
	//int tid = threadIdx.x;	// ֻ��һ���߳̿飬ͨ���س����������ݽ�������
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// �߳̿�block���߳�thread�Ķ�ά��֯��ʽ
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
		tid += blockDim.x * gridDim.x;		// ÿ�ε���������ΪGPU�����߳���
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
	// ������һЩ��ֵ����ֹԽ��
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	// a b���Ƶ�GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// ����������ִ��add()�е��豸����
	// �����˲����߳̿������ΪN
	//add << <N, 1 >> > (dev_a, dev_b, dev_c);
	add << <128, 128 >> > (dev_a, dev_b, dev_c);	// ���䲢���߳̿������Ϊ128��ÿһ���߳̿�����128���߳�

	// c���Ƶ�CPU
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

	// �ͷ�GPU�ڴ�
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}