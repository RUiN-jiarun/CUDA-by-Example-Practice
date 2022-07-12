// ������ʹ���㿽�������ڴ���ʵ�ֵ������
// �عˣ�ʹ��cudaHostAlloc()����ҳ�����ڴ�����֤���ύ���������ڴ�
// ʹ��cudaHosyAllocMapped������ʹ��CUDA�˺�����ֱ�ӷ��������ڴ�
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "book.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float* a, float* b, float* c)
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

/// <summary>
/// �����ڴ�汾
/// </summary>
/// <param name="size"></param>
/// <returns></returns>
float malloc_test(int size)
{
    cudaEvent_t start, stop;
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    for (int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

    dot << <blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));

    // ����CPU������
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    free(a);
    free(b);
    free(partial_c);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("Value calculated:  %f\n", c);

    return elapsedTime;
}

/// <summary>
/// �㿽���ڴ�汾
/// </summary>
/// <param name="size"></param>
/// <returns></returns>
float cuda_host_alloc_test(int size)
{
    cudaEvent_t start, stop;
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // ʹ��cudaHostAllocMapped��־��������ʱ��GPU�з�������ڴ�
    // �������뻺������ʹ��cudaHostAllocWriteCombined��־����ϲ�ʽд�������GPU��ȡ�ڴ�����
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float),
        cudaHostAllocMapped));

    // ����cudaHostGetDevicePointer()�������ڴ���GPU�ϵ���Чָ��
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

    for (int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));

    dot << <blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaThreadSynchronize());
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFreeHost(a));      // ע��ʹ��cudaFreeHost()�ͷ��ڴ�
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(partial_c));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("Value calculated:  %f\n", c);

    return elapsedTime;
}

int main(void)
{
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    if (prop.canMapHostMemory != 1)
    {
        printf("Device can not map memory.\n");
        return 0;
    }

    float elapsedTime;

    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    elapsedTime = malloc_test(N);
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);

    elapsedTime = cuda_host_alloc_test(N);
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);
}