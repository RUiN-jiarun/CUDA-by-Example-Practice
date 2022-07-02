#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
// 为什么要比较?
// 启动的线程块数量最多是32个，如果对于较短的代码，应该采用较小的线程块数量
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c)
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

    // 接下来要将cache中的所有的值相加，需要一个线程来读取保存在cache中的值
    // 但是我们必须保证读取的时候，所有对cache的写入都已经完成
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

int main(void)
{
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    // 为输入数组a、b和输出数组c分配主机内存和设备你存
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    // 填充输入数组并复制到设备上
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    // 调用核函数，指定线程格中线程块的数量和每个块中的线程数量
    dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    // 将c复制到主机上
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