#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"


void showDevice()
{
    cudaDeviceProp prop;

    int cnt;
    HANDLE_ERROR(cudaGetDeviceCount(&cnt));
    for (int i = 0; i < cnt; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("   --- General Information for device %d ---\n", i);
        printf("Name:  %s\n", prop.name);
        printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
        printf("Clock rate:  %d\n", prop.clockRate);
        printf("Device copy overlap:  ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execution timeout :  ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("   --- Memory Information for device %d ---\n", i);
        printf("Total global mem:  %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem:  %ld\n", prop.totalConstMem);
        printf("Max mem pitch:  %ld\n", prop.memPitch);
        printf("Texture Alignment:  %ld\n", prop.textureAlignment);

        printf("   --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count:  %d\n",
            prop.multiProcessorCount);
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n",
            prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2]);
        printf("\n");
    }
}

void setDevice()
{
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;         // 查找主版本号大于1
    prop.minor = 3;         // 且次版本号大于等于3
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    HANDLE_ERROR(cudaSetDevice(dev));
}

int main()
{
    showDevice();
    setDevice();
}