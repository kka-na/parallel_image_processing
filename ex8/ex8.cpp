#include "ex8.h"
#include <stdio.h>
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/include/device_launch_parameters.h>
#include "opencv2/opencv.hpp"

#include <chrono>

using namespace cv;

std::string types[3] = {"Global", "Shared", "Constant"};

void Seq_Gaborfilter(float Gvar, float Gtheta, float Glambda, float Gpsi, int Gkernel_size, float *Gkernel)
{

    if (Gkernel_size % 2 == 0)
        Gkernel_size++;

    for (int x = -Gkernel_size / 2; x <= Gkernel_size / 2; x++)
    {
        for (int y = -Gkernel_size / 2; y <= Gkernel_size / 2; y++)
        {
            int index = (x + Gkernel_size / 2) * Gkernel_size + (y + Gkernel_size / 2);
            Gkernel[index] = exp(-((x * x) + (y * y)) / (2 * Gvar)) * cos(Glambda * (x * cos(Gtheta) + y * sin(Gtheta)) + Gpsi);
        }
    }
}

int main()
{
    Mat pInput = imread("../image/Grab_Image.bmp", 0);
    resize(pInput, pInput, Size(512, 512));

    int w = pInput.cols;
    int ws = pInput.cols;
    int h = pInput.rows;

    float *pDst = new float[w * h];
    Mat pfInput;
    pInput.convertTo(pfInput, CV_32FC1);

    double ptime[3];
    Mat presult[3];

    for (int i = 0; i < 3; i++)
    {
        auto s1 = std::chrono::high_resolution_clock::now();
        float *pcuSrc;
        float *pcuDst;
        float *pcuGkernel;
        // Allocate cuda device memory
        (cudaMalloc((void **)&pcuSrc, w * h * sizeof(float)));
        (cudaMalloc((void **)&pcuDst, w * h * sizeof(float)));

        // copy input image across to the device
        (cudaMemcpy(pcuSrc, pfInput.data, w * h * sizeof(float), cudaMemcpyHostToDevice));

        int kernel_size = 17;
        float *Gkernel = new float[kernel_size * kernel_size];
        Seq_Gaborfilter(0.5, (180.0 * 3.141593 / 180), (0.55), (90 * 3.141593 / 180), kernel_size, Gkernel);

        (cudaMalloc((void **)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
        (cudaMemcpy(pcuGkernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

        // gpu_Gabor
        gpu_Gabor(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size, i);

        // Copy the marker data back to the host
        (cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));

        auto f1 = std::chrono::high_resolution_clock::now();
        ptime[i] = double(std::chrono::duration_cast<std::chrono::microseconds>(f1 - s1).count()) / 1000000;
        Mat result(Size(pInput.cols, pInput.rows), CV_32FC1, pDst);
        result.convertTo(result, CV_8UC1);
        presult[i] = result;

        cudaFree(pcuSrc);
        cudaFree(pcuDst);
    }

    for (int i = 0; i < 3; i++)
        printf("%s Memory Processing Time(8bit): %lf sec\n", types[i].c_str(), ptime[i]);
    for (int i = 1; i < 3; i++)
        printf("%s more Faster than Global about %d % \n", types[i].c_str(), int((ptime[0] / ptime[i]) * 100));
    for (int i = 0; i < 3; i++)
        imwrite("../result/" + types[i] + ".jpg", presult[i]);

    imshow(types[1], presult[1]);
    waitKey(0);

    // for (int i = 0; i < 3; i++)
    // {
    //     imshow(types[i], presult[i]);
    //     waitKey(0);
    // }

    return 0;
}