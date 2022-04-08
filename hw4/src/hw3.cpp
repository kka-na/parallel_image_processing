#include "MyInterpolation.h"
#include <chrono>

int decidePixel();
void showType(int);
void doInterpolation(MyInterpolation, int, int);

string interpolation_type[4] = {"Bilinear", "BiCubic", "Lagrange", "BSpline"};

int main()
{
    cout << "\nParallel Image Processing Programming HW4\n22212231 김가나\n";

    int question_num = 0;

    cout << "\n\n፨ Enter Question number to To check the answer to the problem. ፨" << endl;
    cout << "[ Question #1 ]\n"
         << interpolation_type[0] << endl;
    cout << "[ Question #2 ]\n"
         << interpolation_type[1] << endl;
    cout << "[ Question #3 ]\n"
         << interpolation_type[2] << endl;
    cout << "[ Question #4 ]\n"
         << interpolation_type[3] << endl;

    Mat image = imread("../image/lena512.bmp", 1);
    resize(image, image, Size(80, 80));

    MyInterpolation mi(image);
    int npixel = 0;

    while (true)
    {
        cout << "\n፨ Quit to Enter 0 ፨" << endl;
        cout << "▶▷▶ ";
        cin >> question_num;
        if (question_num != 0)
            npixel = decidePixel();
        showType(question_num);
        if (question_num >= 1 && question_num <= 4)
            doInterpolation(mi, question_num, npixel);
        if (question_num == 0)
        {
            cout << "End Homewrok 4 ... " << endl
                 << endl;
            break;
        }
        else
        {
            cout << "Enter Number again" << endl;
        }
        waitKey(0);
        destroyAllWindows();
    }
    return 0;
}

void showType(int num)
{
    if (num > 0 && num <= 4)
    {
        cout << "\n[ " << interpolation_type[num - 1] << " ]" << endl;
    }
}

void doInterpolation(MyInterpolation mi, int question_num, int npixel)
{

    if (question_num > 0 && question_num <= 4)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        Mat dst1 = mi.SerialInterpolation(question_num, npixel);
        auto finish1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);

        auto start2 = std::chrono::high_resolution_clock::now();
        Mat dst2 = mi.OMPInterpolation(question_num, npixel);
        auto finish2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(finish2 - start2);

        cout << "  |- Processing Time" << endl;
        cout << "     - Serial : " << float(duration1.count()) / 1000000 << " sec" << endl;
        cout << "     - OMP    : " << float(duration2.count()) / 1000000 << " sec" << endl;

        // imwrite("../result/dst.png", mi.dst);
        imwrite("../result/" + interpolation_type[question_num - 1] + "_dst.png", dst1);
        hconcat(mi.dst, dst1, dst1);
        imshow(interpolation_type[question_num - 1] + " Interpolation", dst1);
    }
}

int decidePixel()
{
    int pixel_num;
    cout << "፨ Enter number of pixels for interpolation ፨" << endl;
    cout << "▶▷▶ ";
    cin >> pixel_num;
    return pixel_num;
}