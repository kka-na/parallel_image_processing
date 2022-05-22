#include "mySerial.cpp"
#include "myOpenMP.cpp"
#include "myOpenCV.cpp"
#include "myIPP.cpp"
#include "myCUDA.cpp"

#include <chrono>

int main()
{
    cout << "\nParallel Image Processing Programming HW7\n22212231 김가나\n";
    cout << "\nGaussian Filtering Using Serial, OpenMP, OpenCV, IPP, CUDA\n";
    cout << "Image Size = 4096x4096, Kernel Size = 25x25 , Sigma Value = 25.0f\n\n";

    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(4096, 4096));
    int ksize = 25;
    float sigma = 25.0f;

    // Serial Gaussian Filtering
    auto s1 = std::chrono::high_resolution_clock::now();
    Mat serial = serialGaussian(src, ksize, sigma);
    auto f1 = std::chrono::high_resolution_clock::now();
    float d1 = float(std::chrono::duration_cast<std::chrono::microseconds>(f1 - s1).count()) / 1000000;

    // OpenMP Gaussian Filtering
    auto s2 = std::chrono::high_resolution_clock::now();
    Mat omp = ompGaussian(src, ksize, sigma);
    auto f2 = std::chrono::high_resolution_clock::now();
    float d2 = float(std::chrono::duration_cast<std::chrono::microseconds>(f2 - s2).count()) / 1000000;

    // OpenCV Gaussian Filtering
    auto s3 = std::chrono::high_resolution_clock::now();
    Mat cv = cvGaussian(src, ksize, sigma);
    auto f3 = std::chrono::high_resolution_clock::now();
    float d3 = float(std::chrono::duration_cast<std::chrono::microseconds>(f3 - s3).count()) / 1000000;

    // IPP Gaussian Filtering
    auto s4 = std::chrono::high_resolution_clock::now();
    Mat ipp = ippGaussian(src, ksize, sigma);
    auto f4 = std::chrono::high_resolution_clock::now();
    float d4 = float(std::chrono::duration_cast<std::chrono::microseconds>(f4 - s4).count()) / 1000000;

    // CUDA Gaussian Filtering
    auto s5 = std::chrono::high_resolution_clock::now();
    Mat cuda = cudaGaussian(src, ksize, sigma);
    auto f5 = std::chrono::high_resolution_clock::now();
    float d5 = float(std::chrono::duration_cast<std::chrono::microseconds>(f5 - s5).count()) / 1000000;

    cout << "  |- Processing Time" << endl;
    cout << "   - Serial       : " << d1 << " sec" << endl;
    cout << "   - OpenMP       : " << d2 << " sec" << endl;
    cout << "   - OpenCV       : " << d3 << " sec" << endl;
    cout << "   - IPP          : " << d4 << " sec" << endl;
    cout << "   - CUDA         : " << d5 << " sec" << endl;

    // imshow("Original", src);
    // imshow("Serial", serial);
    // imshow("OpenMP", omp);
    // imshow("OpenCV", cv);
    // imshow("IPP", ipp);
    // imshow("CUDA", cuda);

    imwrite("../result/original.png", src);
    imwrite("../result/serial.png", serial);
    imwrite("../result/omp.png", omp);
    imwrite("../result/cv.png", cv);
    imwrite("../result/ipp.png", ipp);
    imwrite("../result/cuda.png", cuda);

    // waitKey(0);
    // destroyAllWindows();
    return 0;
}
