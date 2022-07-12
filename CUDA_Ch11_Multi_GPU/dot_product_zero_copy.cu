// 本例将使用零拷贝主机内存来实现点积运算
// 回顾：使用cudaHostAlloc()分配页锁定内存来保证不会交换出物理内存
// 使用cudaHosyAllocMapped参数来使得CUDA核函数中直接访问主机内存
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
    __shared__ float cache[threadsPerBlock];        // 设置一个共享内存缓冲区cache，保存每个线程计算的和的值
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // 设置cache中相应位置上的值
    cache[cacheIndex] = temp;

    // 对线程块中的线程进行同步
    __syncthreads();
    // 确保线程块中的每个线程都执行完__syncthreads()前面的语句才会往后执行

    // 进行规约运算（reduction）
    // 每个线程将cache中的两个值相加，将结果再保存回cache，即每次结果数量减半
    // 256个线程需要8次迭代规约为1个值
    // 要求threadPerBlock必须是2的指数
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

    // 迭代的最后保存的值再cache的第一个元素，把他保存到全局内存
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

/// <summary>
/// 主机内存版本
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

    // 结束CPU段运算
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
/// 零拷贝内存版本
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

    // 使用cudaHostAllocMapped标志告诉运行时从GPU中访问这块内存
    // 对于输入缓冲区，使用cudaHostAllocWriteCombined标志代表合并式写入以提高GPU读取内存性能
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float),
        cudaHostAllocMapped));

    // 调用cudaHostGetDevicePointer()获得这块内存在GPU上的有效指针
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

    HANDLE_ERROR(cudaFreeHost(a));      // 注意使用cudaFreeHost()释放内存
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