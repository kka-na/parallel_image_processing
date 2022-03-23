#include "SaltandPepper.h"
#include "Histogram.h"
#include "BlobDetection.h"

int main()
{
    cout << "\nParallel Image Processing Programming HW1\n22212231 김가나\n";

    int question_num = 0;

    cout << "\n\n፨ Enter Question number to To check the answer to the problem. ፨" << endl;
    cout << "[ Question #1 ]\nRemove the Salt and Pepper noise.\n";
    cout << "[ Question #2 ]\nDevelop the tresholding program.(Draw the histogram for each channel of color image) \n";
    cout << "[ Question #3 ]\nDesign an Circle Detection.\n";

    while (true)
    {
        cout << "\n፨ Quit to Enter 0 ፨" << endl;
        cout << "▶▷▶ ";
        cin >> question_num;
        if (question_num == 1)
        { // Salt and Pepper
            SaltandPepper sp;
            Mat src = imread("../image/hw1_1.ppm", IMREAD_GRAYSCALE);
            Mat dst;
            sp.myMedian(src, dst, Size(9, 9));
            hconcat(src, dst, dst);
            imshow("Salt and Pepper", dst);
            imwrite("../result/salt_and_pepper_9x9result.jpg", dst);
        }
        else if (question_num == 2)
        { // Histogram
            Histogram h;
            Mat src = imread("../image/hw1_2.jpg");
            Mat src_g = imread("../image/hw1_2.jpg", IMREAD_GRAYSCALE);

            Mat dst1 = h.GetColorHistogram(src);
            hconcat(src, dst1, dst1);
            imshow("Histogram", dst1);
            imwrite("../result/color_histogram_result.jpg", dst1);
            waitKey(0);

            cout << "For Manual : Enter threshold value " << endl;
            int th_m[3];
            cout << "Threshold B, G, R : ";
            cin >> th_m[0] >> th_m[1] >> th_m[2];
            cout << "Threshold Gray : " << (th_m[0] + th_m[1] + th_m[2]) / 3 << endl;
            Mat dst2 = h.ManualThresholding(src, th_m);
            Mat dst2g = h.ManualThresholding(src_g, th_m);
            cvtColor(dst2g, dst2g, COLOR_GRAY2BGR);

            cout << "For Automatical : Auto Calculated " << endl;
            Mat dst3 = h.AutomaticThresholding(src);
            Mat dst3g = h.AutomaticThresholding(src_g);
            int *th_a = h.GetAutomaticThreshold(1);
            int th_ag = h.GetAutomaticThreshold(0)[0];
            cout << "Threshold B, G, R : ";
            cout << th_a[0] << " " << th_a[1] << " " << th_a[2] << endl;
            cout << "Threshold Gray : " << th_ag << endl;
            cvtColor(dst3g, dst3g, COLOR_GRAY2BGR);

            hconcat(src, dst2, dst2);
            hconcat(dst2, dst2g, dst2);
            hconcat(src, dst3, dst3);
            hconcat(dst3, dst3g, dst3);
            imshow("Manual Thresholding", dst2);
            imwrite("../result/manual_thresholding_result.jpg", dst2);
            imshow("Automatical Thresholding", dst3);
            imwrite("../result/automatic_thresholding_result.jpg", dst3);
        }
        else if (question_num == 3)
        { // Blob Detection
            BlobDetection bd;
            vector<Mat> dsts;
            static string file_names[] = {"../image/circle_blob1.png", "../image/circle_blob2.png", "../image/circle_blob3.png", "../image/circle_blob4.png"};
            for (int i = 0; i < 4; i++)
            {
                Mat src = imread(file_names[i]);
                Mat dst = bd.cvBlobDetection(src);
                hconcat(src, dst, dst);
                dsts.push_back(dst);
            }
            imshow("Blob Detection blob1", dsts[0]);
            imwrite("../result/blob_detection_1.jpg", dsts[0]);
            imshow("Blob Detection blob2", dsts[1]);
            imwrite("../result/blob_detection_2.jpg", dsts[1]);
            imshow("Blob Detection blob3", dsts[2]);
            imwrite("../result/blob_detection_3.jpg", dsts[2]);
            imshow("Blob Detection blob4", dsts[3]);
            imwrite("../result/blob_detection_4.jpg", dsts[3]);
        }
        else if (question_num == 0)
        {
            cout << "End Homewrok 1 ... " << endl
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