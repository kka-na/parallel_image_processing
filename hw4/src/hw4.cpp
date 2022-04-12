#include "Display.h"
#include "Detection.h"

string display_type[4] = {"Capture", "Display", "Face Detection", "Human Detection"};

int main()
{
    cout << "\nParallel Image Processing Programming HW4\n22212231 김가나\n";
    cout << "\nMultiple Video Processing \n";
    for (int i = 0; i < 4; i++)
    {
        cout << " |-" << display_type[i] << endl;
    }
    Display dp;
    Detection dt;

    string svideo = "../video/Endgame.mp4";
    VideoCapture video(svideo);
    if (!video.isOpened())
        return -1;

    namedWindow(display_type[1], 1);
    moveWindow(display_type[1], 1920, 0);
    namedWindow(display_type[2], 1);
    moveWindow(display_type[2], 2400, 0);
    namedWindow(display_type[3], 1);
    moveWindow(display_type[3], 2880, 0);

    Mat frame;
    video >> frame;

#pragma omp parallel sections
    {
#pragma omp section
        {
            printf("Capture Thread #%d\n", omp_get_thread_num());
            dp.CaptureVideo(video, frame);
        }
#pragma omp section
        {

            printf("Display Thread #%d\n", omp_get_thread_num());
            dp.DisplayVideo(frame, display_type[1]);
        }
#pragma omp section
        {
            printf("Face Detection Thread #%d\n", omp_get_thread_num());
            dt.FaceDetection(frame, display_type[2]);
        }
#pragma omp section
        {
            printf("Human Detection Thread #%d\n", omp_get_thread_num());
            dt.HumanDetection(frame, display_type[3]);
        }
    }

    destroyAllWindows();
    return 0;
}
