#include "MyFilter.h"
#include <chrono>

int decidePixel();
void showType(int);

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

    Mat image = imread("../image/lena512.bmp", 0);

    // Timer
    auto start1 = std::chrono::high_resolution_clock::now();
    auto finish1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);
    cout << "Processing Time : " << float(duration1.count()) / 1000000 << " sec" << endl;

    while (true)
    {
        cout << "\n፨ Quit to Enter 0 ፨" << endl;
        cout << "▶▷▶ ";
        cin >> question_num;
        showType(question_num - 1);
        int npixel = decidePixel();
        if (question_num == 1)
        {
        }
        else if (question_num == 2)
        {
        }
        else if (question_num == 3)
        {
        }
        else if (question_num == 4)
        {
        }
        else if (question_num == 0)
        {
            cout << "End Homewrok 4 ... " << endl
                 << endl;
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
    cout << "<" << interpolation_type[num] << ">" << endl;
}

int decidePixel()
{
    int pixel_num;
    cout << "\n፨ Enter number of pixels for interpolation ፨" << endl;
    cout << "▶▷▶ ";
    cin >> pixel_num;
    return pixel_num;
}