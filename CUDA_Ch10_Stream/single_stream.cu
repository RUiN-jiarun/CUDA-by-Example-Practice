#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)


/// <summary>
/// �˺���������a������ֵ��b������ֵ��ƽ��ֵ
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <returns></returns>
__global__ void kernel(int* a, int* b, int* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void)
{
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    // ѡ��֧���豸�ص����ܵ��豸������ִ��һ��CUDA�˺�����ͬʱ���������豸������֮��ִ�и��Ʋ���
    if (!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t stream;
    int* host_a, * host_b, * host_c;
    int* dev_a, * dev_b, * dev_c;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // ��ʼ����
    HANDLE_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // ��������ʹ�õ�ҳ�����ڴ�
    // ��ʹ��cudaHostAlloc()���������ϵĹ̶��ڴ�
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // �����뻺��������Ϊ��С�Ŀ飬����ÿ������ִ��
    // ������������ѭ����ÿ�����ݿ��СΪN
    for (int i = 0; i < FULL_DATA_SIZE; i += N)
    {
        // �������ڴ����첽��ʽ���Ƶ��豸��
        // cudaMemcpy()��ͬ����ʽִ�У�����������ʱ�����Ʋ����Ѿ���ɣ�����������������а����˸��ƽ�ȥ������
        // cudaMemcpyAsync()ֻ��һ������ͨ������streamָ��������������ʱֻ��ȷ����ֵ�����ᱻ����һ���������еĲ���֮ǰִ��
        // ֻ�����첽��ʽ��ҳ�����ڴ���и��Ʋ���
        HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

        kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

        // �����ݴ��豸���Ƶ������ڴ�
        HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));

    }

    // ����������ҳ�����ڴ渴�Ƶ������ڴ�
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    HANDLE_ERROR(cudaEventRecord(stop, 0));

    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    HANDLE_ERROR(cudaStreamDestroy(stream));

    return 0;
}

// 42.8ms