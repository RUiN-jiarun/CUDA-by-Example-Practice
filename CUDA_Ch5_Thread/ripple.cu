#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f

/// <summary>
/// �˺���
/// </summary>
/// <param name="ptr">ָ�򱣴��������ֵ���豸�ڴ�</param>
/// <param name="ticks">����ʱ��</param>
/// <returns></returns>
__global__ void kernel(unsigned char* ptr, int ticks)
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
	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

struct DataBlock
{
	unsigned char* dev_bitmap;
	CPUAnimBitmap* bitmap;
};

/// <summary>
/// �����µ�һ֡
/// </summary>
/// <param name="d"></param>
/// <param name="ticks"></param>
void generate_frame(DataBlock* d, int ticks)
{
	dim3 blocks(DIM / 16, DIM / 16);	// �̸߳��а����Ĳ����߳̿�����
	dim3 threads(16, 16);				// ÿ���߳̿��а������߳�����
	// ����һ�����ɵ�ͼ�������ͼ����DIM*DIM�����أ�ÿ���߳̿���16*16���߳����飬����Ҫ����DIM/16*DIM/16���߳̿�
	kernel << <blocks, threads >> > (d->dev_bitmap, ticks);	// ���ú˺�����������ֵ

	HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

// �ͷ�GPU�Ϸ�����ڴ�
void cleanup(DataBlock* d)
{
	HANDLE_ERROR(cudaFree(d->dev_bitmap));
}

int main(void)
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

	// ִ���豸���룬���ڴ����ݽ��м���
	// ��һ��ָ��generate_frame()�ĺ���ָ�봫�ݸ�anim_and_exit()��ÿ��Ҫ����һ֡�µĶ������͵���generate_frame()����
	bitmap.anim_and_exit((void (*)(void*, int))generate_frame, (void (*)(void*))cleanup);
}
