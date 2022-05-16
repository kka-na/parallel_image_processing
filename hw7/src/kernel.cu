#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_timer.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_cuda.h>

/*
[HOST]
- Image Load
[DEVICE]
- Allocating Device Memory
- Memory Transfer Host to Device
- Applying Image Processing
- Memory transfer Device to Host
[HOST]
- Displaying Image
*/

__global__ void cuda_Filter2D(float *pcuSrc, int w, int h, float *Gkernel, int kernel_size, float *pcuDst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * w + x;
    for (int j = 0; j < w; j++)
    {
        for (int i = 0; i < h; i++)
        {
            pcuDst[index] += pcuSrc[(y + j) * w + (x + i)] * Gkernel[j * kernel_size + i];
        }
    }
}

void gpuGaussian(float *pcuSrc, float *pcuDst, int w, int h, float *Gkernel, int kernel_size)
{
    dim3 grid = dim3(w / 16, h / 16);
    dim3 block = dim3(16, 16, 1);
    cuda_Filter2D<<<grid, block>>>(pcuSrc, w, h, Gkernel, kernel_size, pcuDst);
    cudaThreadSynchronize();
}

double _gaussian2D(float c, float r, double sigma)
{
    return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

void doGaussian(float *pSrc, float *pDst, int w, int h, int kernel_size)
{
    float *pcuSrc;
    float *pcuDst;
    float *Gkernel = new float[kernel_size * kernel_size];
    (cudaMalloc((void **)&pcuSrc, w * h * sizeof(float)));
    (cudaMalloc((void **)&pcuDst, w * h * sizeof(float)));
    (cudaMemcpy(pcuSrc, pSrc, w * h * sizeof(float), cudaMemcpyHostToDevice));

    float *pKernel = new float[kernel_size * kernel_size];
    double sigma = 1.0f;
    for (int c = 0; c < kernel_size; c++)
    {
        for (int r = 0; r < kernel_size; r++)
        {
            Gkernel[r * kernel_size + c] = (float)_gaussian2D((float)(c - kernel_size / 2), (float)(r - kernel_size / 2), sigma);
        }
    }
    (cudaMalloc((void **)&Gkernel, kernel_size * kernel_size * sizeof(float)));
    (cudaMemcpy(pKernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    gpuGaussian(pcuSrc, pcuDst, w, h, Gkernel, kernel_size);
    (cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));
}

// int timer_test(void)
// {
//     StopWatchInterface *timer = NULL;
//     sdkCreateTimer(&timer);
//     sdkStartTimer(&timer);
//     sdkStopTimer(&timer);
//     double time = sdkGetTimerValue(&timer);
//     sdkDeleteTimer(&timer);
//     printf("Processing Time %fsec\n", time);
//     return 0;
// }