#include "ex7.h"
#include <stdio.h>
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/include/device_launch_parameters.h>
#include "opencv2/opencv.hpp"

using namespace cv;

// extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
// extern "C" __global__ void addKernel(int *c, const int *a, const int *b);

// extern "C" void gpu_Gabor(float *pcuSrc, float *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

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
    namedWindow("input", 0);
    namedWindow("output", 0);
    imshow("input", pInput);
    int w = pInput.cols;
    int ws = pInput.cols;
    int h = pInput.rows;

    printf("%d\t%d\t%d\n", h, w, ws);

    // float *pfInput = new float[w*h];
    float *pDst = new float[w * h];
    Mat pfInput;
    pInput.convertTo(pfInput, CV_32FC1);

    /*for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++)
        {
            pSrc[j*w+i] =unsigned char(pInput->imageData[j*ws+i]) ;
        }
    }*/

    // time checker
    clock_t before, now;

    before = clock();

    float *pcuSrc;
    float *pcuDst;
    float *pcuGkernel;
    // Allocate cuda device memory
    (cudaMalloc((void **)&pcuSrc, w * h * sizeof(float)));
    (cudaMalloc((void **)&pcuDst, w * h * sizeof(float)));

    // copy input image across to the device
    (cudaMemcpy(pcuSrc, pfInput.data, w * h * sizeof(float), cudaMemcpyHostToDevice));

    int kernel_size = 5;
    float *Gkernel = new float[kernel_size * kernel_size];
    Seq_Gaborfilter(0.5, (180.0 * 3.141593 / 180), (0.55), (90 * 3.141593 / 180), kernel_size, Gkernel);

    (cudaMalloc((void **)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
    (cudaMemcpy(pcuGkernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    gpu_Gabor(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size);

    // Copy the marker data back to the host
    (cudaMemcpy(pDst, pcuDst, w * h * sizeof(float), cudaMemcpyDeviceToHost));

    now = clock();
    printf("Processing Time(8bit): %lf msec\n", (double)(now - before));
    Mat imgd1(Size(pInput.cols, pInput.rows), CV_32FC1, pDst);
    // Mat imgd2(Size(pInput.rows, pInput.cols), CV_16UC1, output);
    Mat dstdiplay;
    // normalize(imgd1, dstdiplay, 255, 0, NORM_MINMAX, CV_8UC1);
    imgd1.convertTo(dstdiplay, CV_8UC1);
    imshow("output", dstdiplay);
    waitKey(0);
    // time checker
    // tEnd = cvGetTickCount();// for check processing
    // ProTime = 0.001 * (tEnd -tStart) / cvGetTickFrequency(); // for msec
    // printf("Processing time : %f msec\n",ProTime);

    // CvSize cvsize1 = {w ,h};
    // IplImage* TempImage1 = cvCreateImage( cvsize1, IPL_DEPTH_32F, 1);
    // IplImage* TempImage1 = cvCreateImage( cvsize1, IPL_DEPTH_8U, 1);

    //	IplImage* dst = cvCreateImage( cvsize1, IPL_DEPTH_8U, 1);

    //// copy to OpenCV
    //	for (int y = 0; y < cvsize1.height; y++) {
    //		for (int x = 0; x < cvsize1.width; x++) {
    //
    //			cvSetReal2D(TempImage1, y, x, pDst[y*cvsize1.width+x]);
    //	}
    //}

    ////cvSaveImage("result.bmp",TempImage1);
    // cvNamedWindow( "sGFilteredimage1",0);       // ������ ����
    // cvShowImage( "sGFilteredimage1", TempImage1 );// �̹����� ������

    // cvWaitKey(0);
    //  free the device memory

    cudaFree(pcuSrc);
    cudaFree(pcuDst);

    return 0;
}