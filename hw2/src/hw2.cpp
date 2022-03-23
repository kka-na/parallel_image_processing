#include "myIPP.h"
#include "myOpenCV.h"
#include <ctime>

void TestMultipleImages(Mat src, int ptype, int ftype, bool _show);

Size sizes[4] = {Size(256, 256), Size(512, 512), Size(1024, 1024), Size(2048, 2048)};
string processing_type[2] = {"IPP", "OpenCV"};
string filter_type[2] = {"Gaussian", "Median"};

int main()
{
    cout << "\nParallel Image Processing Programming HW2\n22212231 김가나\n";

    int question_num = 0;

    cout << "\n\n፨ Enter Question number to To check the answer to the problem. ፨" << endl;
    cout << "[ Question #1 ]\nImplement Gaussian filter using IPP.\n";
    cout << "[ Question #2 ]\nImpelment Median filter using IPP.\n";
    cout << "[ Question #3 ]\nTest on different size of images 256, 512, 1024, 2048.\n";
    cout << "[ Question #4 ]\nCompare result images, processing time between OpenCV and IPP.\n";

    Mat image = imread("../image/Grab_Image.bmp", 0);

    clock_t start, finish;
    double duration;

    MyIPP mi;
    MyOpenCV mc;

    while (true)
    {
        cout << "\n፨ Quit to Enter 0 ፨" << endl;
        cout << "▶▷▶ ";
        cin >> question_num;

        if (question_num == 1)
        {
            Mat src = image.clone();
            int size = 1;
            int ksize = 5;
            float sigma = 70.f;
            resize(src, src, sizes[size]);
            start = clock();
            Mat dst = mi.IppGaussian(src, sizes[size].width, sizes[size].height, ksize, sigma);
            finish = clock();
            duration = (double)(finish - start) / CLOCKS_PER_SEC;
            cout << "IPP Processing time : " << duration << " sec" << endl;
            hconcat(src, dst, dst);
            imshow("IPP 5x5 Gaussian Filter", dst);
        }
        else if (question_num == 2)
        {
            Mat src = image.clone();
            int size = 1;
            int ksize = 5;
            resize(src, src, sizes[size]);
            start = clock();
            Mat dst = mi.IppMedian(src, sizes[size].width, sizes[size].height, ksize);
            finish = clock();
            duration = (double)(finish - start) / CLOCKS_PER_SEC;
            cout << "IPP Processing time : " << duration << " sec" << endl;
            hconcat(src, dst, dst);
            imshow("IPP 5x5 Median Filter", dst);
        }
        else if (question_num == 3)
        {
            cout << "Using IPP" << endl;
            Mat src = image.clone();
            cout << "[ Checking Gaussian ]" << endl;
            TestMultipleImages(src, 0, 0, true);
            cout << "[ Checking Medain ]" << endl;
            TestMultipleImages(src, 0, 1, true);
        }
        else if (question_num == 4)
        {
            cout << "Using OpenCV" << endl;
            Mat src = image.clone();
            cout << "[ Checking Gaussian ]" << endl;
            TestMultipleImages(src, 1, 0, true);
            cout << "[ Checking Medain ]" << endl;
            TestMultipleImages(src, 1, 1, true);
        }
        else if (question_num == 0)
        {
            cout << "End Homewrok 2 ... " << endl
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

void TestMultipleImages(Mat src, int ptype, int ftype, bool _show)
{
    MyIPP mi;
    MyOpenCV mc;

    clock_t start, finish;
    double duration;
    int ksize = 5;
    float sigma = 70.f;
    vector<Mat> dsts;

    string ptypes = processing_type[ptype];
    string ftypes = filter_type[ftype];

    for (int i = 0; i < 4; i++)
    {
        Mat ssrc;
        Mat dst;
        resize(src, ssrc, sizes[i]);
        if (ftype == 0)
        {
            if (ptype == 0)
            {
                start = clock();
                dst = mi.IppGaussian(ssrc, sizes[i].width, sizes[i].height, ksize, sigma);
                finish = clock();
            }
            else if (ptype == 1)
            {
                start = clock();
                dst = mc.CvGaussian(ssrc, ksize, sigma);
                finish = clock();
            }
        }
        else if (ftype == 1)
        {
            if (ptype == 0)
            {
                start = clock();
                dst = mi.IppMedian(ssrc, sizes[i].width, sizes[i].height, ksize);
                finish = clock();
            }
            else if (ptype == 1)
            {
                start = clock();
                dst = mc.CvMedian(ssrc, ksize);
                finish = clock();
            }
        }
        duration = (double)(finish - start) / CLOCKS_PER_SEC;

        cout << ptypes + "Processing time of Image " << sizes[i] << " : " << duration << " sec" << endl;
        // hconcat(ssrc, dst, dst);
        dsts.push_back(dst);
    }
    if (_show)
    {

        for (int i = 0; i < 4; i++)
        {
            string size = "[" + to_string(sizes[i].width) + " x " + to_string(sizes[i].width) + "]";
            // imshow(ptypes + " 5x5 " + ftypes + " Filter on Image [" + size + "]", dsts[i]);
            imwrite("../result/" + ptypes + "5x5" + ftypes + size + ".jpg", dsts[i]);
        }
    }
}