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

    cudaStream_t stream0, stream1;
    int* host_a, * host_b, * host_c;
    int* dev_a0, * dev_b0, * dev_c0;
    int* dev_a1, * dev_b1, * dev_c1;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // 初始化流
    HANDLE_ERROR(cudaStreamCreate(&stream0));
    HANDLE_ERROR(cudaStreamCreate(&stream1));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

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
    // 这里有两个流，我们要按照CUDA引擎的原理合理调度并高效使用
    // 保证依赖性：复制c在执行核函数之后
    // 同时采用宽度优先，stream0对c的复制不会阻塞steam1对a、b的复制
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        // 将复制a的操作放入stream0和stream1的队列
        HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

        // 将复制b的操作放入stream0和stream1的队列
        HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

        // 将核函数的执行放入stream0和stream1的队列
        kernel << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
        kernel << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);

        // 将复制c的操作放入stream0和stream1的队列
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));

    }

    // 将计算结果从页锁定内存复制到主机内存
    HANDLE_ERROR(cudaStreamSynchronize(stream0));
    HANDLE_ERROR(cudaStreamSynchronize(stream1));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a0));
    HANDLE_ERROR(cudaFree(dev_b0));
    HANDLE_ERROR(cudaFree(dev_c0));
    HANDLE_ERROR(cudaFree(dev_a1));
    HANDLE_ERROR(cudaFree(dev_b1));
    HANDLE_ERROR(cudaFree(dev_c1));
    HANDLE_ERROR(cudaStreamDestroy(stream0));
    HANDLE_ERROR(cudaStreamDestroy(stream1));


    return 0;
}

// 37.7ms