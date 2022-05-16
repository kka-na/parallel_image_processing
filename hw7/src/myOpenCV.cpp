#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat cvGaussian(Mat src, int _ksize)
{
    Mat dst(src.size(), src.type());
    cv::GaussianBlur(src, dst, Size(_ksize, _ksize), 70.0f);
    return dst;
}