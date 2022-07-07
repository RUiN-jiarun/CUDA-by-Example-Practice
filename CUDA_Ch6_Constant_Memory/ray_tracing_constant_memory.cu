/* 这个例子中使用了常量内存。常量内存用于保存在核函数执行期间不会发生变化的数据。*/
/* 用常量内存啦i替换全局内存可以有效减少内存带宽。*/
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
/// 定义球体的结构
/// </summary>
struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;

	/// <summary>
	/// 判断光线是否与球体相较
	/// </summary>
	/// <param name="ox">光线起点像素x值</param>
	/// <param name="oy">光线起点像素y值</param>
	/// <param name="n"></param>
	/// <returns>相机到光线命中球面处的距离</returns>
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

// 本例中的输入数据就是球面数组，可以将其保存到常量内存
// 同时他也应该是一个全局变量
__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char* ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	// 把图像坐标偏移(DIM / 2)，使得z轴穿过图像中心
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	// 每条光线要判断与球面相交的情况
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		// 获得交点距离相机距离，并更新保存最接近的球面和颜色值
		// 注意，这个值将保存在输出图像的缓冲区中
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

	// 判断相交情况后，将当前颜色保存到输出图像
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}


int main()
{
	// 记录起始时间
	cudaEvent_t start, stop;
	// 创建cuda事件
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	
	//Sphere* s;
	

	// 在GPU上分配内存以计算输出位图
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	// 为Sphere数据集分配内存
	// HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// 随机生成球面中心坐标、颜色及半径
	// 分配临时内存，对其初始化，并复制到GPU上的内存，然后释放临时内存
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
	//HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));		// 这个函数的意思是复制到常量内存
	free(temp_s);

	// 生成位图
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (dev_bitmap);

	// 复制到CPU以显示
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// 获取渲染时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	HANDLE_ERROR(cudaFree(dev_bitmap));

	bitmap.display_and_exit();

}