#ifndef MyOpenCV_H
#define MyOpenCV_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class MyOpenCV
{
public:
    Mat CvGaussian(Mat src, int _ksize, float _sigma)
    {
        Mat dst(src.size(), src.type());
        cv::GaussianBlur(src, dst, Size(_ksize, _ksize), _sigma);
        return dst;
    }
    Mat CvMedian(Mat src, int _ksize)
    {
        Mat dst(src.size(), src.type());
        cv::medianBlur(src, dst, _ksize);
        return dst;
    }
};

#endif