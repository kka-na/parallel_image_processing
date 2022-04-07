#ifndef MyFilter_h
#define MyFilter_h

#include <omp.h>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class MyFilter
{
public:
    void ompGaussian(const Mat &srcImg, Mat &dstImg, Size size)
    {
        Mat kn = Mat::zeros(size, CV_32FC1);
        double sigma = 1.0;
        float *kn_data = (float *)kn.data;
        for (int c = 0; c < kn.cols; c++)
        {
            for (int r = 0; r < kn.rows; r++)
            {
                kn_data[r * kn.cols + c] =
                    (float)gaussian2D((float)(c - kn.cols / 2),
                                      (float)(r - kn.rows / 2), sigma);
            }
        }
        ompKernelConv(srcImg, dstImg, kn);
    }

    void ompKernelConv(const Mat &srcImg, Mat &dstImg, const Mat &kn)
    {
        dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
        int wd = srcImg.cols;
        int hg = srcImg.rows;
        int kwd = kn.cols;
        int khg = kn.rows;
        int rad_w = kwd / 2;
        int rad_h = khg / 2;

        float *kn_data = (float *)kn.data;
        uchar *srcData = (uchar *)srcImg.data;
        uchar *dstData = (uchar *)dstImg.data;

        float wei, tmp, sum;

#pragma omp parallel for reduction(+ \
                                   : wei, tmp, sum)
        for (int c = rad_w + 1; c < wd - rad_w; c++)
        {
            for (int r = rad_h + 1; r < hg - rad_h; r++)
            {
                tmp = 0.f;
                sum = 0.f;
                for (int kc = -rad_w; kc <= rad_w; kc++)
                {
                    for (int kr = -rad_h; kr <= rad_h; kr++)
                    {
                        wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
                        tmp += wei * (float)srcData[(r + kr) * wd + (c + kc)];
                        sum += wei;
                    }
                }
                if (sum != 0.f)
                    tmp = abs(tmp) / sum;
                else
                    tmp = abs(tmp);
                if (tmp > 255.f)
                    tmp = 255.f;
                dstData[r * wd + c] = (uchar)tmp;
            }
        }
    }

    void serialGaussian(const Mat &srcImg, Mat &dstImg, Size size)
    {
        Mat kn = Mat::zeros(size, CV_32FC1);
        double sigma = 1.0;
        float *kn_data = (float *)kn.data;
        for (int c = 0; c < kn.cols; c++)
        {
            for (int r = 0; r < kn.rows; r++)
            {
                kn_data[r * kn.cols + c] =
                    (float)gaussian2D((float)(c - kn.cols / 2),
                                      (float)(r - kn.rows / 2), sigma);
            }
        }
        serialKernelConv(srcImg, dstImg, kn);
    }

    void serialKernelConv(const Mat &srcImg, Mat &dstImg, const Mat &kn)
    {
        dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
        int wd = srcImg.cols;
        int hg = srcImg.rows;
        int kwd = kn.cols;
        int khg = kn.rows;
        int rad_w = kwd / 2;
        int rad_h = khg / 2;

        float *kn_data = (float *)kn.data;
        uchar *srcData = (uchar *)srcImg.data;
        uchar *dstData = (uchar *)dstImg.data;

        float wei, tmp, sum;

        for (int c = rad_w + 1; c < wd - rad_w; c++)
        {
            for (int r = rad_h + 1; r < hg - rad_h; r++)
            {
                tmp = 0.f;
                sum = 0.f;
                for (int kc = -rad_w; kc <= rad_w; kc++)
                {
                    for (int kr = -rad_h; kr <= rad_h; kr++)
                    {
                        wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
                        tmp += wei * (float)srcData[(r + kr) * wd + (c + kc)];
                        sum += wei;
                    }
                }
                if (sum != 0.f)
                    tmp = abs(tmp) / sum;
                else
                    tmp = abs(tmp);
                if (tmp > 255.f)
                    tmp = 255.f;
                dstData[r * wd + c] = (uchar)tmp;
            }
        }
    }

    double gaussian2D(float c, float r, double sigma)
    {
        return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
    }
};

#endif