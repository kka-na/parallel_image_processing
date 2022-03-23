
#include <iostream>
#include "opencv2/opencv.hpp"
#include <ipp.h>

using namespace cv;
using namespace std;

int main()
{
    cout << "\nParallel Image Processing Programming\n22212231 김가나\n";
    Mat image = imread("../image/Grab_Image.bmp", 0);
    resize(image, image, image.size() / 6);
    IppiSize size, tsize;
    size.width = image.cols;
    size.height = image.rows;
    Ipp8u *S_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);
    // Ipp8u *D_img = (Ipp8u *)ippsMalloc_8u((size.width - 1) * (size.height - 1));
    Ipp8u *T_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);
    Ipp16s *T = (Ipp16s *)ippsMalloc_16s(size.width * size.height);
    ippiCopy_8u_C1R((const Ipp8u *)image.data, size.width, S_img, size.width, size);
    tsize.width = image.cols;
    tsize.height = image.rows;
    ippiFilterSobelHorizBorder_8u16s_C1R(S_img, size.width, T, size.width * 2, tsize, ippMskSize5x5, ippBorderConst, 255, T_img);
    Size s;
    s.width = image.cols;
    s.height = image.rows;
    cv::Mat dst(s, CV_16S, T);
    imshow("Image", image);
    imshow("DST", dst);
    waitKey(0);
    destroyAllWindows();
    return 0;
}