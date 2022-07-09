#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "book.h"
#include "cpu_bitmap.h"
#include "cuda.h"
#include "cuda_gl_interop.h"

#define DIM 512

PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;


// 定义两个全局变量来保存句柄，只想在OpenGL和CUDA之间共享的数据
GLuint bufferObj;					// OpenGL
cudaGraphicsResource* resource;		// CUDA

__global__ void kernel(uchar4* ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x / (float)DIM - 0.5f;
    float fy = y / (float)DIM - 0.5f;
    unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

    // 访问uchar4类型而非unsigned char*
    ptr[offset].x = 0;
    ptr[offset].y = green;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}

static void key_func(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);
        exit(0);
    }
}

static void draw_func(void)
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}


int main(int argc, char** argv)
{
    cudaDeviceProp  prop;
    int dev;

    // 确保选择一个拥有高于1.0版本的GPU
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    // 知道CUDA设备的ID，并把ID dev传递进去
    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    // 调用GLUT来初始化OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow("bitmap");

    glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // 创建共享数据缓冲区
    glGenBuffers(1, &bufferObj);                            // 生成缓冲区句柄
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);    // 将句柄绑定到像素缓冲区
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB); // 分配DIM*DIM个32位值，最后一个参数表示将被修改
    // 为bufferObj注册图形资源
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

    // CUDA创建指向缓冲区的句柄，需要设备内存中的实际地址传递给核函数
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));     // 映射共享资源
    uchar4* devPtr;
    size_t  size;
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));   // 请求一个指向被映射资源的指针

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel << <grids, threads >> > (devPtr);                // 核函数生成图像数据
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));   // 取消映射，确保应用程序的CUDA部分和图形部分之间同步

    // 设置GLUT并启动绘制循环
    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMainLoop();
}