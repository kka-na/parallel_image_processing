#include "SerialInterpolation.h"
#include "OMPInterpolation.h"

#include <chrono>

float doSerial();
float doOpenMP();

int main()
{
    cout << "\nParallel Image Processing Programming Project1\n22212231 김가나\n";
    cout << "\nBayer Interpolation Using OpenMP & Comparesz with Serial\n";

    float serial = doSerial();
    float omp = doOpenMP();

    cout << "  |- Processing Time" << endl;
    cout << "     - Serial    : " << serial << " sec" << endl;
    cout << "     - OpenMP    : " << omp << " sec" << endl;

    destroyAllWindows();
    return 0;
}

float doSerial()
{
    SerialInterpolation si;
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = si.doBayer();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    resize(dst, dst, Size(dst.cols / 3, dst.rows / 3));
    imshow("Bayer Interpolation ( Serial Processing )", dst);
    // imwrite("../result/serial.png", dst);
    waitKey(0);

    return float(duration.count()) / 1000000;
}

float doOpenMP()
{
    OMPInterpolation oi;
    auto start = std::chrono::high_resolution_clock::now();
    Mat dst = oi.doBayer();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    resize(dst, dst, Size(dst.cols / 3, dst.rows / 3));
    imshow("Bayer Interpolation ( OpenMP Processing )", dst);
    // imwrite("../result/omp.png", dst);
    waitKey(0);

    return float(duration.count()) / 1000000;
}