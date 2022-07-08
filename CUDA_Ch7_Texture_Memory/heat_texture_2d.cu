#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "cuda.h"
#include "book.h"
#include "cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// 使用GPU纹理内存加速
texture<float, 2>  texConstSrc;
texture<float, 2>  texIn;
texture<float, 2>  texOut;


/// <summary>
/// 步骤1：给定包含初始输入温度的网格，将其中作为热源的单元温度值给到相应单元
/// 从纹理内存读取而非全局内存
/// </summary>
/// <param name="iptr"></param>
/// <returns></returns>
__global__ void copy_const_kernel(float* iptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc, x, y);
    if (c)
        iptr[offset] = c;
}

/// <summary>
/// 步骤2：更新计算输出温度网格
/// 使用特殊的函数，不再从缓冲区中读，而是读取请求转发到纹理内存
/// </summary>
/// <param name="dst"></param>
/// <param name="dstOut">使用哪个缓冲区作为输入和输出</param>
/// <returns></returns>
__global__ void blend_kernel(float* dst, bool dstOut)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   
        left++;
    if (x == DIM - 1) 
        right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)   
        top += DIM;
    if (y == DIM - 1) 
        bottom -= DIM;

    float t, l, c, r, b;
    if (dstOut)
    {
        t = tex2D(texIn, x, y - 1);
        l = tex2D(texIn, x - 1, y);
        c = tex2D(texIn, x, y);
        r = tex2D(texIn, x + 1, y);
        b = tex2D(texIn, x, y + 1);
    }
    else
    {
        t = tex2D(texOut, x, y - 1);
        l = tex2D(texOut, x - 1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x + 1, y);
        b = tex2D(texOut, x, y + 1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

/// <summary>
/// 更新函数中需要的全局变量
/// </summary>
struct DataBlock
{
    unsigned char* output_bitmap;
    float* dev_inSrc;
    float* dev_outSrc;
    float* dev_constSrc;
    CPUAnimBitmap* bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};


void anim_gpu(DataBlock* d, int ticks)
{
    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap* bitmap = d->bitmap;

    // boolean标志，选择每次迭代中哪个是输入/输出
    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++)
    {
        /*copy_const_kernel << <blocks, threads >> > (d->dev_inSrc, d->dev_constSrc);
        blend_kernel << <blocks, threads >> > (d->dev_outSrc, d->dev_inSrc);
        swap(d->dev_inSrc, d->dev_outSrc);*/
        float* in, * out;
        if (dstOut)
        {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else
        {
            in = d->dev_outSrc;
            out = d->dev_inSrc;
        }
        copy_const_kernel << <blocks, threads >> > (in);
        blend_kernel << <blocks, threads >> > (out, dstOut);
        dstOut = !dstOut;       // 更改boolean值来切换输入与输出缓冲区
    }

    float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_inSrc);

    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Average Time per frame:  %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock* d)
{
    // 解绑纹理内存
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);

    HANDLE_ERROR(cudaFree(d->dev_inSrc));
    HANDLE_ERROR(cudaFree(d->dev_outSrc));
    HANDLE_ERROR(cudaFree(d->dev_constSrc));

    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main()
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));
    
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));

    // 假设float为4字符，即rgba
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));

    // 绑定纹理内存到之前生命的纹理引用
    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, DIM, DIM, sizeof(float) * DIM);
    HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_inSrc, DIM, DIM, sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_outSrc, DIM, DIM, sizeof(float) * DIM));

    // 初始化静态数据
    float* temp = (float*)malloc(bitmap.image_size());
    for (int i = 0; i < DIM * DIM; i++)
    {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++)
    {
        for (int x = 400; x < 500; x++)
        {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

    // 初始化输入数据
    for (int y = 800; y < DIM; y++)
    {
        for (int x = 0; x < 200; x++)
        {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
    free(temp);

    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu, (void (*)(void*))anim_exit);
}

