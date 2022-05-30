#include <stdio.h>
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/include/device_launch_parameters.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_timer.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_cuda.h>
int timer_test(void)
{
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    int a = 0;
    for (int i = 0; i < 100; i++)
    {
        a += 1;
    }
    sdkStopTimer(&timer);
    double time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    printf("Processing Time %fsec\n", time);
    return 0;
}

__global__ void cuda_Filter2D(float *pSrcImage, int SrcWidth, int SrcHeight, float *pKernel, int KWidth, int KHeight, float *pDstImage)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * SrcWidth + x;
    int border;
    float temp;

    //	input[index] = clamp1(input);
    if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
    {
        temp = 0;
        for (int j = 0; j < KHeight; j++)
        {
            for (int i = 0; i < KWidth; i++)
            {
                int iindex = (j + y) * SrcWidth + (i + x);
                temp += pSrcImage[iindex] * pKernel[j * KWidth + i];
            }
        }
        pDstImage[index] = temp;
    }
    else
    {
        pDstImage[index] = 0;
    }
}
/*

__global__
void Gaussian(double* In, double* Out, int kDim, int inWidth, int outWidth, int outHeight) {
    extern __shared__ double loadIn[];

    // trueDim is tile dimension without halo cells
    int trueDimX = blockDim.x - (kDim-1);
    int trueDimY = blockDim.y - (kDim-1);

    // trueDim used in place of blockDim so Grid step/stride does not consider halo cells
    int col = (blockIdx.x * trueDimX) + threadIdx.x;
    int row = (blockIdx.y * trueDimY) + threadIdx.y;

    if (col < outWidth && row < outHeight) { // Filter out-of-bounds threads

        // Load input tile into shared memory for the block
        loadIn[threadIdx.y * blockDim.x + threadIdx.x] = In[row * inWidth + col];
        __syncthreads();

        if (threadIdx.y < trueDimY && threadIdx.x < trueDimX) { // Filter extra threads used for halo cells
            double acc = 0;
            for (int i = 0; i < kDim; ++i)
                for (int j = 0; j < kDim; ++j)
                    acc += loadIn[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * K[(i * kDim) + j];
            Out[row * inWidth + col] = acc;
        }
    } else
        loadIn[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
}
*/
void gpu_Gabor(float *pcuSrc, float *pcuDst, int w, int h, float *cuGkernel, int kernel_size, int type)
{

    dim3 block = dim3(16, 16);
    dim3 grid = dim3(w / 16, h / 16);

    if (type == 0)
    {
        cuda_Filter2D<<<grid, block>>>(pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
        cudaThreadSynchronize();
    }
    else if (type == 1)
    {
        cuda_Filter2D<<<grid, block, sizeof(float) * kernel_size * kernel_size>>>(pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
        cudaThreadSynchronize();
    }
    else if (type == 2)
    {
        float constKernel[kernel_size * kernel_size];
        cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float) * kernel_size * kernel_size);
        cuda_Filter2D<<<grid, block>>>(pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
    }
}