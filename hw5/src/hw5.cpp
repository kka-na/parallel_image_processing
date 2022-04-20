#include "SSETest.h"
#include "SerialMean.h"
#include "SSEMean.h"
#include <chrono>

float doSerialMean();
float doSSEMean();
void doTest();

int main()
{
    cout << "\nParallel Image Processing Programming HW5\n22212231 김가나\n";
    cout << "\nMean Filtering Using SSE & Compare with Serial\n";

    float serial = doSerialMean();
    float sse = doSSEMean();

    cout << "  |- Processing Time" << endl;
    cout << "     - Serial    : " << serial << " sec" << endl;
    cout << "     - SSE       : " << sse << " sec" << endl;

    destroyAllWindows();
    return 0;
}

float doSerialMean()
{
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(512, 512));
    SerialMean sm(src);
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = sm.moreOptimizeMean();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    hconcat(src, dst, dst);
    imshow("3x3 Mean Filter Test ( Serial Processing )", dst);
    imwrite("../result/serial_mean.jpg", dst);
    waitKey(0);

    return float(duration.count()) / 1000000;
}

float doSSEMean()
{
    // int data[6][6] = {1, 4, 0, 1, 3, 1,
    //                    2, 2, 4, 2, 2, 3,
    //                    1, 0, 1, 0, 1, 0,
    //                    1, 2, 1, 0, 2, 2,
    //                    2, 5, 3, 1, 2, 5,
    //                    1, 1, 4, 2, 3, 0};
    //  Mat src(6, 6, CV_8UC1, data);
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(512, 512));
    SSEMean sm(src);
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = sm.myMean();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    hconcat(src, dst, dst);
    imshow("3x3 Mean Filter Test ( SSE Processing )", dst);
    imwrite("../result/sse_mean.jpg", dst);
    waitKey(0);
    return float(duration.count()) / 1000000;
}

void doTest()
{
    SSETest ts;

    short A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    short B[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    short C[8] = {0};
    short D[8] = {0};

    // C Program
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 8; i++)
    {
        C[i] = A[i] + B[i];
    }
    printf("%d, %d, %d, %d, %d, %d, %d, %d\n", C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);

    auto finish1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);

    // SIMD Program
    auto start2 = std::chrono::high_resolution_clock::now();
    ts.SIMDTest(A, B, D);
    auto finish2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(finish2 - start2);

    cout << "  |- Processing Time" << endl;
    cout << "     - C    : " << float(duration1.count()) / 1000000 << " sec" << endl;
    cout << "     - SIMD : " << float(duration2.count()) / 1000000 << " sec" << endl;
}