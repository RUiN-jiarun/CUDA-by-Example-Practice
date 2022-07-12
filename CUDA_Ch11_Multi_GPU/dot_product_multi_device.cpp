// ������ʹ�ö�GPU����
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N / 2 + threadsPerBlock - 1) / threadsPerBlock);    // ע������

__global__ void dot(int size, float* a, float* b, float* c)
{
    __shared__ float cache[threadsPerBlock];        // ����һ�������ڴ滺����cache������ÿ���̼߳���ĺ͵�ֵ
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // ����cache����Ӧλ���ϵ�ֵ
    cache[cacheIndex] = temp;

    // ���߳̿��е��߳̽���ͬ��
    __syncthreads();
    // ȷ���߳̿��е�ÿ���̶߳�ִ����__syncthreads()ǰ������Ż�����ִ��

    // ���й�Լ���㣨reduction��
    // ÿ���߳̽�cache�е�����ֵ��ӣ�������ٱ����cache����ÿ�ν����������
    // 256���߳���Ҫ8�ε�����ԼΪ1��ֵ
    // Ҫ��threadPerBlock������2��ָ��
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

    // ��������󱣴��ֵ��cache�ĵ�һ��Ԫ�أ��������浽ȫ���ڴ�
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

/// <summary>
/// ����������ʱ���豸��ʶ����������С�Լ�����ָ�����뻺������ָ��
/// ����һ������������
/// </summary>
struct DataStruct
{
    int deviceID;
    int size;
    float* a;
    float* b;
    float returnValue;
};

/// <summary>
/// GPU���д���
/// ע�⣬����һ��void*������������void*����������ʵ�������̺߳���
/// </summary>
/// <param name="pvoidData"></param>
/// <returns></returns>
void* routine(void* pvoidData)
{
    DataStruct* data = (DataStruct*)pvoidData;
    HANDLE_ERROR(cudaSetDevice(data->deviceID));

    int size = data->size;
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    a = data->a;
    b = data->b;
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

    dot << <blocksPerGrid, threadsPerBlock >> > (size, dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    free(partial_c);

    data->returnValue = c;
    return 0;
}


int main(void)
{
    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
    {
        printf("We need at least two compute 1.0 or greater "
            "devices, but only found %d\n", deviceCount);
        return 0;
    }

    float* a = (float*)malloc(sizeof(float) * N); HANDLE_NULL(a);
    float* b = (float*)malloc(sizeof(float) * N); HANDLE_NULL(b);

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // ֻ��������GPUΪ��
    DataStruct data[2];
    data[0].deviceID = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].size = N / 2;
    data[1].a = a + N / 2;
    data[1].b = b + N / 2;

    CUTThread thread = start_thread(routine, &(data[0]));
    routine(&(data[1]));
    end_thread(thread);

    free(a);
    free(b);

    printf("Value calculated:  %f\n", data[0].returnValue + data[1].returnValue);

    return 0;
}