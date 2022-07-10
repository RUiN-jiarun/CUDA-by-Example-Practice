// 这个程序的作用是测试cudaMemcpy()在可分配内存和页锁定上的性能
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#define SIZE    (64*1024*1024)

/// <summary>
/// 使用malloc()分配可分页主机内存
/// </summary>
/// <param name="size"></param>
/// <param name="up"></param>
/// <returns></returns>
float cuda_malloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int* a, * dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int*)malloc(size * sizeof(*a));
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    // 为size个证书分别分配主机缓冲区和GPU缓冲区，执行100次复制操作，计时
    HANDLE_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < 100; i++)
    {
        if (up)
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

/// <summary>
/// 使用cudaHostAlloc()分配固定内存
/// </summary>
/// <param name="size"></param>
/// <param name="up"></param>
/// <returns></returns>
float cuda_host_alloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int* a, * dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // 与malloc()使用方式相同
    // 默认分配固定主机内存
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < 100; i++)
    {
        if (up)
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    HANDLE_ERROR(cudaFreeHost(a));      // 清除内存
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}


int main(void)
{
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;


    // try it with cudaMalloc
    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using cudaMalloc:  %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cudaHostAlloc:  %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));
}