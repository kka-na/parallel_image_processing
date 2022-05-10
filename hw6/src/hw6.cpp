#include "HW5Mean.h"
#include "HW6Mean.h"
#include <chrono>

float doHW5Mean();
float doHW6Mean();

int main()
{
    cout << "\nParallel Image Processing Programming HW6\n22212231 김가나\n";
    cout << "\nMean Filtering Using SSE with 2 Methods\n";

    float hw5 = doHW5Mean();
    float hw6 = doHW6Mean();

    cout << "  |- Processing Time" << endl;
    cout << "   - HW5       : " << hw5 << " sec" << endl;
    cout << "   - HW6       : " << hw6 << " sec" << endl;

    destroyAllWindows();
    return 0;
}

float doHW5Mean()
{
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(512, 512));
    HW5Mean sm(src);
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = sm.myMean();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    hconcat(src, dst, dst);
    imshow("3x3 Mean Filter Test ( HW5 Method )", dst);
    imwrite("../result/hw5_mean.jpg", dst);
    waitKey(0);
    return float(duration.count()) / 1000000;
}

float doHW6Mean()
{
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(512, 512));
    HW6Mean sm(src);
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = sm.myMean();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    hconcat(src, dst, dst);
    imshow("3x3 Mean Filter Test ( HW6 Methods )", dst);
    imwrite("../result/hw6_mean.jpg", dst);
    waitKey(0);
    return float(duration.count()) / 1000000;
}
