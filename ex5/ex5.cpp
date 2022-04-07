#include <omp.h>
#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int DisplayVideo(string, string);
int GaussianVideo(string, string);
int GaborVideo(string, string);
int SobelVideo(string, string);
int CannyVideo(string, string);

int main()
{
    cout << "\nParallel Image Processing Programming Excersise\n22212231 김가나\n";
    string video = "../video/Endgame.mp4";

#pragma omp parallel sections
    {
#pragma omp section
        DisplayVideo(video, "Display Video");

#pragma omp section
        GaussianVideo(video, "Gaussian Video");

#pragma omp section
        GaborVideo(video, "Gabor Video");

#pragma omp section
        SobelVideo(video, "Sobel Video");

#pragma omp section
        CannyVideo(video, "Canny Video");
    }
    destroyAllWindows();
    return 0;
}

int DisplayVideo(string _video, string windowName)
{
    VideoCapture video(_video);
    if (!video.isOpened())
        return -1;
    double fstart, fend, fps;
    namedWindow(windowName, 1);
    for (;;)
    {
        fstart = omp_get_wtime();
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }
        resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
        fend = omp_get_wtime();
        fps = (1 / (fend - fstart));
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        imshow(windowName, frame);
        waitKey(30);
    }
    return 0;
}

int GaussianVideo(string _video, string windowName)
{
    VideoCapture video(_video);
    if (!video.isOpened())
        return -1;
    double fstart, fend, fps;
    namedWindow(windowName, 1);
    for (;;)
    {
        fstart = omp_get_wtime();
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }
        resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
        GaussianBlur(frame, frame, Size(9, 9), 0, 0, 4);
        fend = omp_get_wtime();
        fps = (1 / (fend - fstart));
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        imshow(windowName, frame);
        waitKey(30);
    }
    return 0;
}

int GaborVideo(string _video, string windowName)
{
    VideoCapture video(_video);
    if (!video.isOpened())
        return -1;
    double fstart, fend, fps;
    namedWindow(windowName, 1);
    for (;;)
    {
        fstart = omp_get_wtime();
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }
        // cvtColor(frame, frame, COLOR_RGBA2GRAY);
        resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
        Mat kernel = getGaborKernel(Size(9, 9), 1, 0, 1.0, 0.02, 0);
        filter2D(frame, frame, CV_8UC3, kernel);
        fend = omp_get_wtime();
        fps = (1 / (fend - fstart));
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        imshow(windowName, frame);
        waitKey(30);
    }
    return 0;
}

int SobelVideo(string _video, string windowName)
{
    VideoCapture video(_video);
    if (!video.isOpened())
        return -1;
    double fstart, fend, fps;
    namedWindow(windowName, 1);
    for (;;)
    {
        fstart = omp_get_wtime();
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }
        cvtColor(frame, frame, COLOR_RGBA2GRAY);
        resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
        Sobel(frame, frame, CV_8U, 1, 1);
        fend = omp_get_wtime();
        fps = (1 / (fend - fstart));
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        imshow(windowName, frame);
        waitKey(30);
    }
    return 0;
}

int CannyVideo(string _video, string windowName)
{
    VideoCapture video(_video);
    if (!video.isOpened())
        return -1;
    double fstart, fend, fps;
    namedWindow(windowName, 1);
    for (;;)
    {
        fstart = omp_get_wtime();
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            destroyWindow(windowName);
            break;
        }
        cvtColor(frame, frame, COLOR_RGBA2GRAY);
        resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
        Canny(frame, frame, 0, 3, 5);
        fend = omp_get_wtime();
        fps = (1 / (fend - fstart));
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        imshow(windowName, frame);
        waitKey(30);
    }
    return 0;
}

/*
auto start = std::chrono::steady_clock::now();
        auto finish = std::chrono::steady_clock::now();
        float fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        putText(frame, "FPS : " + to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
*/