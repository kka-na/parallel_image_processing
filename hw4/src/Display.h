#ifndef Display_H
#define Display_H

#include <omp.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class Display
{
public:
    int CaptureVideo(VideoCapture video, Mat _frame)
    {
        while (true)
        {
            video >> _frame;
            if (_frame.empty())
                break;
            this_thread::sleep_for(30ms);
        }
        return 0;
    }
    int DisplayVideo(Mat _frame, string windowName)
    {
        double fstart = omp_get_wtime();
        while (true)
        {
            Mat frame = _frame.clone();
            resize(frame, frame, Size(480, 480));
            double fend = omp_get_wtime();
            double fps = (1 / (fend - fstart));
            putText(frame, to_string(fps) + "FPS", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            imshow(windowName, frame);
            if (waitKey(30) == 27)
                break;
            fstart = fend;
        }
        return 0;
    }
};
#endif
