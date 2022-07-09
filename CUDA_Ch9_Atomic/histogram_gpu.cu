#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "device_atomic_functions.hpp"
#include <stdio.h>
#include <stdlib.h>
//#include "cuda.h"
#include "book.h"

#define SIZE    (100*1024*1024)

/// <summary>
/// 使用全局内存原子操作
/// </summary>
/// <param name="buffer"></param>
/// <param name="size"></param>
/// <param name="histo"></param>
/// <returns></returns>
//__global__ void hist_kernel(unsigned char* buffer, long size, unsigned int* hist)
//{
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x;
//    // 每个线程知道起始偏移i和递增的数量，遍历输入数组，递增直方图中相应元素的值
//    while (i < size)
//    {
//        atomicAdd(&hist[buffer[i]], 1);         // CUDA的原子操作，对该位置的值增加1
//        i += stride;
//    }
//}

/// <summary>
/// 使用共享内存原子操作和全局内存原子操作
/// </summary>
/// <param name="buffer"></param>
/// <param name="size"></param>
/// <param name="histo"></param>
/// <returns></returns>
__global__ void hist_kernel(unsigned char* buffer, long size, unsigned int* hist)
{

    // 分配共享内存换种区并初始化
    __shared__  unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();                        // 同步以确保提交最后的写入操作

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size)
    {
        atomicAdd(&temp[buffer[i]], 1);     
        i += stride;
    }
    // 将每个线程块的临时直方图合并到全局缓冲hist[]
    __syncthreads();
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);     // 将线程块的直方图的每个元素都加到最终直方图中相应位置的元素上
}


int main()
{
    unsigned char* buffer = (unsigned char*)big_random_block(SIZE);

    cudaEvent_t  start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    unsigned char* dev_buffer;
    unsigned int* dev_hist;
    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_hist, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_hist, 0, 256 * sizeof(int)));

    // kernel launch - 2x the number of mps gave best timing
    cudaDeviceProp  prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    hist_kernel << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_hist);

    unsigned int hist[256];
    HANDLE_ERROR(cudaMemcpy(hist, dev_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    long histCount = 0;
    for (int i = 0; i < 256; i++)
    {
        histCount += hist[i];
    }
    printf("Histogram Sum:  %ld\n", histCount);

    // 验证与CPU计算结果一致
    for (int i = 0; i < SIZE; i++)
        hist[buffer[i]]--;
    for (int i = 0; i < 256; i++)
    {
        if (hist[i] != 0)
            printf("Failure at %d!  Off by %d\n", i, hist[i]);
    }

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    cudaFree(dev_hist);
    cudaFree(dev_buffer);
    free(buffer);
    return 0;
}