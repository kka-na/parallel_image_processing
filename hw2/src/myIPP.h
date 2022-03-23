#ifndef MYIPP_H
#define MYIPP_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <ipp.h>

using namespace cv;
using namespace std;

class MyIPP
{
public:
    Mat IppSobel(Mat image, int width, int height)
    {
        IppiSize size = {width, height};
        IppiSize roiSize = {width - 1, height - 1};
        int srcStep = width * sizeof(Ipp8u);
        int dstStep = height * sizeof(Ipp16s);
        Ipp8u *pSrc = (Ipp8u *)ippsMalloc_8u(width * height);
        ippiCopy_8u_C1R((const Ipp8u *)image.data, width, pSrc, srcStep, size);
        Ipp16s *pDst = (Ipp16s *)ippsMalloc_16s((width) * (height));
        Ipp8u *pBuffer = (Ipp8u *)ippsMalloc_8u(width * height);
        IppiBorderType borderType = ippBorderConst;
        ippiFilterSobelHorizBorder_8u16s_C1R(pSrc + srcStep, srcStep, pDst, dstStep, roiSize, ippMskSize5x5, borderType, 0, pBuffer);
        Mat dst(image.size(), CV_16S, pDst);
        return dst;
    }
    Mat IppGaussian(Mat image, int width, int height, int _ksize, float _sigma)
    {
        IppiSize size = {width, height};
        IppiSize roiSize = {width, height};
        int srcStep = width * sizeof(Ipp8u);
        int dstStep = height * sizeof(Ipp8u);
        Ipp8u borderValue = 255;
        Ipp32u kernelSize = _ksize;
        Ipp32f sigma = _sigma;
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
    Mat IppMedian(Mat image, int width, int height, int _ksize)
    {
        IppiSize size = {width, height};
        IppiSize roiSize = {width, height};
        int srcStep = width * sizeof(Ipp8u);
        int dstStep = height * sizeof(Ipp8u);
        Ipp8u borderValue = 255;
        IppiSize maskSize = {_ksize, _ksize};
        int iBufSize = 0;
        IppiBorderType borderType = ippBorderConst;
        Ipp8u *pSrc = (Ipp8u *)ippsMalloc_8u(width * height);
        Ipp8u *pDst = (Ipp8u *)ippsMalloc_8u(width * height);
        ippiCopy_8u_C1R((const Ipp8u *)image.data, width, pSrc, srcStep, size);
        Ipp8u *pBuffer = (Ipp8u *)ippsMalloc_8u(width * height);

        ippiFilterMedianBorderGetBufferSize(roiSize, maskSize, ipp8u, 1, &iBufSize);
        pBuffer = ippsMalloc_8u(iBufSize);
        ippiFilterMedianBorder_8u_C1R(pSrc + srcStep, srcStep, pDst, dstStep, roiSize, maskSize, borderType, borderValue, pBuffer);
        Mat dst(image.size(), CV_8U, pDst);
        return dst;
    }
};

#endif