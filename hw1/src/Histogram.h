#ifndef Histogram_h
#define Histogram_h

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class Histogram
{

private:
    int th_a[3];
    int th_ag[3];

public:
    Mat GetGrayHistogram(Mat &src)
    {
        Mat hist;
        float range[] = {0, 256};
        const float *histRange[] = {range};
        int hitSize = 256;

        calcHist(&src, 1, 0, Mat(), hist, 1, &hitSize, histRange);

        int hist_w = 512;
        int hist_h = src.cols;
        int bin_w = cvRound((double)hist_w / hitSize);

        Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
        normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        for (int i = 1; i < hitSize; i++)
        {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
            if (i % 8 == 0)
            {
                putText(histImage, to_string(i), Point((i * 2) - 2, hist_h - 4), 1, 0.5, Scalar::all(255));
            }
        }
        return histImage;
    }
    Mat GetColorHistogram(Mat &src)
    {
        Mat b_hist, g_hist, r_hist;
        vector<Mat> bgr_planes;
        split(src, bgr_planes);
        int histSize = 256;
        float range[] = {0, 256};
        const float *histRange[] = {range};
        bool uniform = true, accumulate = false;

        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

        int hist_w = 512; // 256*2
        int hist_h = src.cols;
        int bin_w = cvRound((double)hist_w / histSize);

        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        for (int i = 1; i < histSize; i++)
        {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                 Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                 Scalar(0, 0, 255), 2, 8, 0);
            if (i % 8 == 0)
            {
                putText(histImage, to_string(i), Point((i * 2) - 2, hist_h - 4), 1, 0.5, Scalar::all(255));
            }
        }

        return histImage;
    }
    Mat ManualThresholding(Mat &src, int ths[3])
    {
        Mat thresholdImage;
        if (src.channels() == 3)
        {
            vector<Mat> splitedImage;
            split(src, splitedImage);
            threshold(splitedImage[0], splitedImage[0], ths[0], 255, THRESH_BINARY);
            threshold(splitedImage[1], splitedImage[1], ths[1], 255, THRESH_BINARY);
            threshold(splitedImage[2], splitedImage[2], ths[2], 255, THRESH_BINARY);
            merge(splitedImage, thresholdImage);
        }
        else
        {
            int th = (ths[0] + ths[1] + ths[2]) / 3;
            threshold(src, thresholdImage, th, 255, THRESH_BINARY);
        }
        return thresholdImage;
    }
    Mat AutomaticThresholding(Mat &src)
    {
        Mat thresholdImage, testImage;
        if (src.channels() == 3)
        {
            vector<Mat> splitedImage, testImage;
            split(src, splitedImage);
            split(src, testImage);

            long double th_b = threshold(splitedImage[0], testImage[0], 0, 255, THRESH_OTSU);
            long double th_g = threshold(splitedImage[1], testImage[1], 0, 255, THRESH_OTSU);
            long double th_r = threshold(splitedImage[2], testImage[2], 0, 255, THRESH_OTSU);
            threshold(splitedImage[0], splitedImage[0], th_b, 255, THRESH_BINARY);
            threshold(splitedImage[1], splitedImage[1], th_g, 255, THRESH_BINARY);
            threshold(splitedImage[2], splitedImage[2], th_r, 255, THRESH_BINARY);
            this->th_a[0] = th_b;
            this->th_a[1] = th_g;
            this->th_a[2] = th_r;
            merge(splitedImage, thresholdImage);
        }
        else
        {
            long double th = threshold(src, testImage, 0, 255, THRESH_OTSU);
            threshold(src, thresholdImage, th, 255, THRESH_BINARY);
            this->th_ag[0] = th;
            this->th_ag[1] = 0;
            this->th_ag[2] = 0;
        }
        return thresholdImage;
    }
    int *GetAutomaticThreshold(int type)
    {
        if (type == 0)
        {
            return this->th_ag;
        }
        else
        {
            return this->th_a;
        }
    }
};

#endif /*Histogram_h*/