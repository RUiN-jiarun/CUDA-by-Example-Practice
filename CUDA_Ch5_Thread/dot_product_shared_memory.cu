#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
// ΪʲôҪ�Ƚ�?
// �������߳̿����������32����������ڽ϶̵Ĵ��룬Ӧ�ò��ý�С���߳̿�����
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c)
{
    __shared__ float cache[threadsPerBlock];        // ����һ�������ڴ滺����cache������ÿ���̼߳���ĺ͵�ֵ
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // ����cache����Ӧλ���ϵ�ֵ
    cache[cacheIndex] = temp;

    // ������Ҫ��cache�е����е�ֵ��ӣ���Ҫһ���߳�����ȡ������cache�е�ֵ
    // �������Ǳ��뱣֤��ȡ��ʱ�����ж�cache��д�붼�Ѿ����
    // ���߳̿��е��߳̽���ͬ��
    __syncthreads();
    // ȷ���߳̿��е�ÿ���̶߳�ִ����__syncthreads()ǰ������Ż�����ִ��

    // ���й�Լ���㣨reduction��
    // ÿ���߳̽�cache�е�����ֵ��ӣ�������ٱ����cache����ÿ�ν����������
    // 256���߳���Ҫ8�ε�����ԼΪ1��ֵ
    // Ҫ��threadPerBlock������2��ָ��
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // 1    2   3   4   5   6   7   8
    // 1+5 2+6 3+7 4+8
    // 1    2   3   4

    // ��������󱣴��ֵ��cache�ĵ�һ��Ԫ�أ��������浽ȫ���ڴ�
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(void)
{
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    // Ϊ��������a��b���������c���������ڴ���豸���
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    // ����������鲢���Ƶ��豸��
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    // ���ú˺�����ָ���̸߳����߳̿��������ÿ�����е��߳�����
    dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    // ��c���Ƶ�������
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c,
        2 * sum_squares((float)(N - 1)));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    free(a);
    free(b);
    free(partial_c);
}