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

__global__ void cuda_Filter2D(float *pcuSrc, int w, int h, float *pcuKernel, int ksize, float *pcuDst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * w + x;
    for (int j = 0; j < ksize; j++)
    {
        for (int i = 0; i < ksize; i++)
        {
            pcuDst[index] += pcuSrc[(y + j) * w + (x + i)] * pcuKernel[j * ksize + i];
        }
    }
}

void gpuGaussian(float *pcuSrc, float *pcuDst, int w, int h, float *pcuKernel, int ksize)
{
    dim3 grid = dim3(w / 16, h / 16);
    dim3 block = dim3(16, 16, 1);
    cuda_Filter2D<<<grid, block>>>(pcuSrc, w, h, pcuKernel, ksize, pcuDst);
    cudaThreadSynchronize();
}

double _gaussian2D(float c, float r, double sigma)
{
    return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

void doGaussian(float *pSrc, float *pDst, int w, int h, int ksize, float _sigma)
{
    float *pcuSrc;
    float *pcuDst;
    float *pKernel = new float[ksize * ksize];
    (cudaMalloc((void **)&pcuSrc, w * h * sizeof(float)));
    (cudaMalloc((void **)&pcuDst, w * h * sizeof(float)));
    (cudaMemcpy(pcuSrc, pSrc, w * h * sizeof(float), cudaMemcpyHostToDevice));

    float *pcuKernel = new float[ksize * ksize];
    double sigma = _sigma;
    float kSum = 0.0;

    for (int j = 0; j < ksize; j++)
    {
        for (int i = 0; i < ksize; i++)
        {
            pKernel[i * ksize + j] = (float)_gaussian2D(float(j - ksize / 2), float(i - ksize / 2), sigma);
            kSum += pKernel[i * ksize + j];
        }
    }
    // Normalize
    for (int j = 0; j < ksize; j++)
    {
        for (int i = 0; i < ksize; i++)
        {
            pKernel[i * ksize + j] /= kSum;
        }
    }

    (cudaMalloc((void **)&pcuKernel, ksize * ksize * sizeof(float)));
    (cudaMemcpy(pcuKernel, pKernel, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

    gpuGaussian(pcuSrc, pcuDst, w, h, pcuKernel, ksize);
    (cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));
}