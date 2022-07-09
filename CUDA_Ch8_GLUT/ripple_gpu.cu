#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "gpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f

/// <summary>
/// 核函数
/// </summary>
/// <param name="ptr">指向保存输出像素值的设备内存</param>
/// <param name="ticks">动画时间</param>
/// <returns></returns>
__global__ void kernel(uchar4* ptr, int ticks)
{
	// 将线程索引映射到像素位置
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// 计算该位置的值
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f *
		cos(d / 10.0f - ticks / 7.0f) /
		(d / 10.0f + 1.0f));
	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

/// <summary>
/// 生成新的一帧
/// </summary>
/// <param name="pixels"></param>
/// <param name=""></param>
/// <param name="ticks"></param>
void generate_frame(uchar4* pixels, void*, int ticks)
{
	dim3 blocks(DIM / 16, DIM / 16);	// 线程格中包含的并行线程块数量
	dim3 threads(16, 16);				// 每个线程块中包含的线程数量
	// 对于一张生成的图像，如果该图像有DIM*DIM个像素，每个线程块有16*16的线程数组，则需要启动DIM/16*DIM/16个线程块
	kernel << <blocks, threads >> > (pixels, ticks);	// 调用核函数计算像素值

}

int main(void)
{
	GPUAnimBitmap bitmap(DIM, DIM, NULL);

	// 执行设备代码，对内存内容进行计算
	// 将一个指向generate_frame()的函数指针传递给anim_and_exit()，每当要生成一帧新的动画，就调用generate_frame()函数
	bitmap.anim_and_exit((void (*)(uchar4*, void*, int))generate_frame, NULL);
}
