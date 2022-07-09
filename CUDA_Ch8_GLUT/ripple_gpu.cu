#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "gpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f

/// <summary>
/// �˺���
/// </summary>
/// <param name="ptr">ָ�򱣴��������ֵ���豸�ڴ�</param>
/// <param name="ticks">����ʱ��</param>
/// <returns></returns>
__global__ void kernel(uchar4* ptr, int ticks)
{
	// ���߳�����ӳ�䵽����λ��
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// �����λ�õ�ֵ
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
/// �����µ�һ֡
/// </summary>
/// <param name="pixels"></param>
/// <param name=""></param>
/// <param name="ticks"></param>
void generate_frame(uchar4* pixels, void*, int ticks)
{
	dim3 blocks(DIM / 16, DIM / 16);	// �̸߳��а����Ĳ����߳̿�����
	dim3 threads(16, 16);				// ÿ���߳̿��а������߳�����
	// ����һ�����ɵ�ͼ�������ͼ����DIM*DIM�����أ�ÿ���߳̿���16*16���߳����飬����Ҫ����DIM/16*DIM/16���߳̿�
	kernel << <blocks, threads >> > (pixels, ticks);	// ���ú˺�����������ֵ

}

int main(void)
{
	GPUAnimBitmap bitmap(DIM, DIM, NULL);

	// ִ���豸���룬���ڴ����ݽ��м���
	// ��һ��ָ��generate_frame()�ĺ���ָ�봫�ݸ�anim_and_exit()��ÿ��Ҫ����һ֡�µĶ������͵���generate_frame()����
	bitmap.anim_and_exit((void (*)(uchar4*, void*, int))generate_frame, NULL);
}
