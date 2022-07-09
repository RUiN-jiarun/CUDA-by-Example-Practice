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


// ��������ȫ�ֱ�������������ֻ����OpenGL��CUDA֮�乲�������
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

    // ����uchar4���Ͷ���unsigned char*
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

    // ȷ��ѡ��һ��ӵ�и���1.0�汾��GPU
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    // ֪��CUDA�豸��ID������ID dev���ݽ�ȥ
    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    // ����GLUT����ʼ��OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow("bitmap");

    glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // �����������ݻ�����
    glGenBuffers(1, &bufferObj);                            // ���ɻ��������
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);    // ������󶨵����ػ�����
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB); // ����DIM*DIM��32λֵ�����һ��������ʾ�����޸�
    // ΪbufferObjע��ͼ����Դ
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

    // CUDA����ָ�򻺳����ľ������Ҫ�豸�ڴ��е�ʵ�ʵ�ַ���ݸ��˺���
    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));     // ӳ�乲����Դ
    uchar4* devPtr;
    size_t  size;
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));   // ����һ��ָ��ӳ����Դ��ָ��

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel << <grids, threads >> > (devPtr);                // �˺�������ͼ������
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));   // ȡ��ӳ�䣬ȷ��Ӧ�ó����CUDA���ֺ�ͼ�β���֮��ͬ��

    // ����GLUT����������ѭ��
    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMainLoop();
}