#include <iostream>
#include "opencv2/opencv.hpp"
#include <ipp.h>

using namespace cv;
using namespace std;

Mat ippGaussian(Mat image, int _ksize)
{
    int width = image.cols;
    int height = image.rows;
    IppiSize size = {width, height};
    IppiSize roiSize = {width, height};
    int srcStep = width * sizeof(Ipp8u);
    int dstStep = height * sizeof(Ipp8u);
    Ipp8u borderValue = 255;
    Ipp32u kernelSize = _ksize;
    Ipp32f sigma = 70.0f;
    IppFilterGaussianSpec *pSpec = NULL;
    int iBufSize = 0, iSpecSize = 0;
    IppiBorderType borderType = ippBorderConst;
    Ipp8u *pSrc = (Ipp8u *)ippsMalloc_8u(width * height);
    Ipp8u *pDst = (Ipp8u *)ippsMalloc_8u(width * height);
    ippiCopy_8u_C1R((const Ipp8u *)image.data, width, pSrc, srcStep, size);
    Ipp8u *pBuffer = (Ipp8u *)ippsMalloc_8u(width * height);

    ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u, 1, &iSpecSize, &iBufSize);
    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
    pBuffer = ippsMalloc_8u(iBufSize);

    ippiFilterGaussianInit(roiSize, kernelSize, sigma, borderType, ipp8u, 1, pSpec, pBuffer);
    ippiFilterGaussianBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, borderValue, pSpec, pBuffer);
    Mat dst(image.size(), CV_8U, pDst);
    return dst;
}