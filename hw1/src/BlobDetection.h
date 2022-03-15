#ifndef BlobDetection_h
#define BlobDetection_h

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

class BlobDetection
{
public:
    Mat cvBlobDetection(Mat srcImg)
    {
        Mat img = srcImg.clone();
        SimpleBlobDetector::Params params;
        params.minThreshold = 10;
        params.maxThreshold = 300;
        params.filterByArea = true;
        params.minArea = 100;
        params.maxArea = 10000;
        params.filterByCircularity = true;
        params.minCircularity = 0.7;
        params.filterByConvexity = true;
        params.minConvexity = 0.97;
        params.filterByInertia = true;
        params.minInertiaRatio = 0.9;
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        std::vector<KeyPoint> keypoints;
        detector->detect(img, keypoints);
        Mat result;
        drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        return result;
    }
};

#endif