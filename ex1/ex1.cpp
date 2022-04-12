
#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main()
{
    cout << "\nParallel Image Processing Programming\n22212231 김가나\n";
    Mat img_cat = imread("../image/cat.jpg", IMREAD_COLOR); // IMREAD_COLOR or IMREAD_GRAYSCALE;

    imshow("Sample", img_cat);
    waitKey(0);
    destroyAllWindows();
    return 0;
}