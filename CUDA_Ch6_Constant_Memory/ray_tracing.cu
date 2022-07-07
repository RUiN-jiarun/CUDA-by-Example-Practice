#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "book.h"
#include "cpu_bitmap.h"

#define rnd(x)	(x * rand() / RAND_MAX)
#define SPHERES 20
#define INF		2e10f
#define DIM		1024

/// <summary>
/// ��������Ľṹ
/// </summary>
struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;

	/// <summary>
	/// �жϹ����Ƿ����������
	/// </summary>
	/// <param name="ox">�����������xֵ</param>
	/// <param name="oy">�����������yֵ</param>
	/// <param name="n"></param>
	/// <returns>����������������洦�ľ���</returns>
	__device__ float hit(float ox, float oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

__global__ void kernel(Sphere* s, unsigned char* ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	// ��ͼ������ƫ��(DIM / 2)��ʹ��z�ᴩ��ͼ������
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	// ÿ������Ҫ�ж��������ཻ�����
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		// ��ý������������룬�����±�����ӽ����������ɫֵ
		// ע�⣬���ֵ�����������ͼ��Ļ�������
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}

	// �ж��ཻ����󣬽���ǰ��ɫ���浽���ͼ��
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}


int main()
{
	// ��¼��ʼʱ��
	cudaEvent_t start, stop;
	// ����cuda�¼�
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	Sphere* s;

	// ��GPU�Ϸ����ڴ��Լ������λͼ
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// ΪSphere���ݼ������ڴ�
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// ������������������ꡢ��ɫ���뾶
	// ������ʱ�ڴ棬�����ʼ���������Ƶ�GPU�ϵ��ڴ棬Ȼ���ͷ���ʱ�ڴ�
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}
	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
	free(temp_s);

	// ����λͼ
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (s, dev_bitmap);

	// ���Ƶ�CPU����ʾ
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// ��ȡ��Ⱦʱ��
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	HANDLE_ERROR(cudaFree(dev_bitmap));
	HANDLE_ERROR(cudaFree(s));

	bitmap.display_and_exit();
}