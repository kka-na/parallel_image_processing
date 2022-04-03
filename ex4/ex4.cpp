#include "MyFilter.h"
#include <chrono>

int main()
{
    cout << "\nParallel Image Processing Programming Excersise\n22212231 김가나\n";
    Mat src = imread("../image/Grab_Image.bmp", IMREAD_GRAYSCALE); // Grab_Image.bmp", IMREAD_GRAYSCALE);
    Mat src1 = src.clone();
    Mat src2 = src.clone();
    Mat dst1, dst2;

    MyFilter mf;

    auto start1 = std::chrono::high_resolution_clock::now();
    mf.serialGaussian(src1, dst1, Size(27, 27));
    auto finish1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);

    auto start2 = std::chrono::high_resolution_clock::now();
    mf.ompGaussian(src2, dst2, Size(27, 27));
    auto finish2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(finish2 - start2);

    cout << "Serial Gaussian Processint Time : " << float(duration1.count()) / 1000000 << " sec" << endl;
    cout << "OpenMP Gaussian Processint Time : " << float(duration2.count()) / 1000000 << " sec" << endl;

    Mat dst;
    hconcat(src, dst1, dst);
    hconcat(dst, dst2, dst);
    resize(dst, dst, Size(dst.cols / 7, dst.rows / 7));
    imshow("Ex4", dst);
    waitKey(0);
    destroyAllWindows();
    return 0;
}