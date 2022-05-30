#include "myCUDA.cpp"

#include <chrono>

int main()
{
    cout << "\nParallel Image Processing Programming HW8\n22212231 김가나\n";
    cout << "\nGaussian Filtering Using CUDA with Vaious Set of Memory Type\n";
    cout << "Global, Shared, Constant\n";
    cout << "Image Size = 4096x4096, Kernel Size = 25x25 , Sigma Value = 25.0f\n\n";

    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(4096, 4096));
    int ksize = 25;
    float sigma = 25.0f;

    // CUDA Global Memory Gaussian Filtering
    auto s0 = std::chrono::high_resolution_clock::now();
    Mat global = cudaGaussian(src, ksize, sigma, 0);
    auto f0 = std::chrono::high_resolution_clock::now();
    float d0 = float(std::chrono::duration_cast<std::chrono::microseconds>(f0 - s0).count()) / 1000000;

    // CUDA Shared Memory Gaussian Filtering
    auto s1 = std::chrono::high_resolution_clock::now();
    Mat shared = cudaGaussian(src, ksize, sigma, 1);
    auto f1 = std::chrono::high_resolution_clock::now();
    float d1 = float(std::chrono::duration_cast<std::chrono::microseconds>(f1 - s1).count()) / 1000000;

    // CUDA Constant Memory Gaussian Filtering
    auto s2 = std::chrono::high_resolution_clock::now();
    Mat constant = cudaGaussian(src, ksize, sigma, 2);
    auto f2 = std::chrono::high_resolution_clock::now();
    float d2 = float(std::chrono::duration_cast<std::chrono::microseconds>(f2 - s2).count()) / 1000000;

    cout << "  |- Processing Time" << endl;
    cout << "   - Global         : " << d0 << " sec" << endl;
    cout << "   - Shared         : " << d1 << " sec" << endl;
    cout << "   - Constant       : " << d2 << " sec" << endl
         << endl;

    cout << "  |- Result Summary" << endl;
    cout << "   - Shared is " << int((d0 / d1) * 100) << "\% faster than Global" << endl;
    cout << "   - Constant is " << int((d0 / d2) * 100) << "\% faster than Global" << endl;

    // imshow("Global", global);
    // imshow("Shared", shared);
    // imshow("Constant", constant);

    imwrite("../result/global.png", global);
    imwrite("../result/shared.png", shared);
    imwrite("../result/constant.png", constant);

    // waitKey(0);
    // destroyAllWindows();
    return 0;
}
