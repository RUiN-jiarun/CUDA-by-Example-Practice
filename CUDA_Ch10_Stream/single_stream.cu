#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)


/// <summary>
/// 核函数，计算a中三个值和b中三个值的平均值
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <returns></returns>
__global__ void kernel(int* a, int* b, int* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void)
{
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    // 选择支持设备重叠功能的设备，即在执行一个CUDA核函数的同时，可以在设备与主机之间执行复制操作
    if (!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t stream;
    int* host_a, * host_b, * host_c;
    int* dev_a, * dev_b, * dev_c;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // 初始化流
    HANDLE_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // 分配由流使用的页锁定内存
    // 即使用cudaHostAlloc()分配主机上的固定内存
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // 将输入缓冲区划分为更小的块，并在每个块上执行
    // 在整体数据上循环，每个数据块大小为N
    for (int i = 0; i < FULL_DATA_SIZE; i += N)
    {
        // 将锁定内存以异步方式复制到设备上
        // cudaMemcpy()以同步方式执行，即函数返回时，复制操作已经完成，并且在输出缓冲区中包含了复制进去的内容
        // cudaMemcpyAsync()只是一个请求，通过参数stream指定流，函数返回时只能确定赋值操作会被当下一个放入流中的操作之前执行
        // 只能用异步方式对页锁定内存进行复制操作
        HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

        kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

        // 将数据从设备复制到锁定内存
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));

    }

    // 将计算结果从页锁定内存复制到主机内存
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    HANDLE_ERROR(cudaStreamDestroy(stream));

    return 0;
}

// 42.8ms