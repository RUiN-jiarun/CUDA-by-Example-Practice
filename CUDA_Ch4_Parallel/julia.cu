#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"
#include "cpu_bitmap.h"

#define DIM 1000

/// <summary>
/// �����Ĵ洢�ṹ
/// </summary>
struct cuComplex
{
	float   r;
	float   i;
	// cuComplex( float a, float b ) : r(a), i(b)  {}
	__device__ cuComplex(float a, float b) : r(a), i(b) {} // Fix error for calling host function from device
	__device__ float magnitude2(void)
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r + a.r, i + a.i);
	}
};


/// <summary>
/// �ж�ĳ�����Ƿ�����Julia��
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns>boolean</returns>
__device__ int julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

/// <summary>
/// CUDA�˺�����
/// </summary>
/// <param name="ptr"></param>
/// <returns></returns>
__global__ void kernel(unsigned char* ptr)
{
	// ��threadIdx/blockIdxӦ��ɫ������λ��
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// �����λ�õ�ֵ
	int juliaVal = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaVal;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}




/// <summary>
/// ����Julia������
/// Z_n+1 = Z_n * Z_n + C
/// </summary>
/// <returns></returns>
int main()
{
	// ͨ��λͼ���ߴ���DIM*DIM��С��ͼ��
	CPUBitmap bitmap(DIM, DIM);
	// ����ָ�뱣���豸�����ݵĸ���
	unsigned char* dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// ������ά���̸߳�
	dim3 grid(DIM, DIM);
	// ����CUDA��ʱ�Զ��ѵ���ά��Сָ��Ϊ1
	kernel << <grid, 1 >> > (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}