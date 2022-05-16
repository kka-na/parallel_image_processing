#include <iostream>
#include <stdio.h>

#include "opencv2/opencv.hpp"

#include "myCUDA.h"

using namespace cv;
using namespace std;

Mat cudaGaussian(Mat src, int _ksize)
{
    Mat dst(src.size(), CV_8UC1);
    int w = src.cols;
    int h = src.rows;

    float *pSrc = new float[w * h];
    float *pDst = new float[w * h];
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            pSrc[j * w + i] = uchar(src.at<uchar>(j * w + i));
        }
    }

    doGaussian(pSrc, pDst, w, h, _ksize);

    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            dst.at<uchar>(j, i) = pDst[j * w + i];
        }
    }

    return dst;
}