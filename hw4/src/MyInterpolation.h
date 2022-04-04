#ifndef MyInterpolation_h
#define MyInterpolation_h

#include <omp.h>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class MyInterpolation
{
public:
    Mat Serial_Interpolation(Mat imgL, Mat imgR, Mat matHomo)
    {
        /*         |_________H________|   |_X'_|
                   | h'11  h'12  h'13 |   | X' |
        inv(H)X' = | h'21  h'22  h'23 | * | Y' | => "Backward Warping ! "
                   | h'31  h'32   1   |   | 1  |
        */
        // Same with Forwarding
        Mat ptsR = allPoint2Mat(imgR);
        Mat ptsAfterHOMO = matHomo * ptsR;
        divide(ptsAfterHOMO.row(0), ptsAfterHOMO.row(2), ptsAfterHOMO.row(0));
        divide(ptsAfterHOMO.row(1), ptsAfterHOMO.row(2), ptsAfterHOMO.row(1));

        pair<int, int> xMinMax = getMinMax(ptsAfterHOMO, 0);
        pair<int, int> yMinMax = getMinMax(ptsAfterHOMO, 1);

        int w, h;
        if (xMinMax.second > imgL.cols)
            w = xMinMax.second; // + xMinMax.first;
        else
            w = imgL.cols; //+ xMinMax.first;
        if (yMinMax.second > imgL.rows)
            h = yMinMax.second; // + yMinMax.first;
        else
            h = imgL.rows; // + xMinMax.first;

        int wWp = xMinMax.second - xMinMax.first;
        int hWp = yMinMax.second - yMinMax.first;

        Mat ptsWP = allPoint2MatbyIdx(xMinMax, yMinMax);
        Mat invHOMO = matHomo.inv();
        Mat ptsAfterInvHOMO = invHOMO * ptsWP;
        divide(ptsAfterInvHOMO.row(0), ptsAfterInvHOMO.row(2), ptsAfterInvHOMO.row(0));
        divide(ptsAfterInvHOMO.row(1), ptsAfterInvHOMO.row(2), ptsAfterInvHOMO.row(1));

        Mat result = Mat::zeros(h, w, imgL.type());

        for (int r = 0; r < hWp; r++)
        {
            for (int c = 0; c < wWp; c++)
            {
                // using Bilinear Interpolation , can get
                double x = double(ptsAfterInvHOMO.at<double>(0, r * wWp + c));
                double y = double(ptsAfterInvHOMO.at<double>(1, r * wWp + c));
                if ((x < imgL.cols - 1) && (y < imgL.rows - 1) && (x > 1) && (y > 1))
                {
                    vector<double> biRGB = Bilinear(imgR, x, y);
                    for (int i = 0; i < 3; i++)
                    {
                        // result.at<Vec3b>(r+2*(yMinMax.first),c+2*(xMinMax.first))[i] = biRGB[i];
                        result.at<Vec3b>(r + yMinMax.first, c + xMinMax.first)[i] = biRGB[i];
                    }
                }
            }
        }
        return result;
    }
    vector<double> Bilinear(Mat imgR, double x, double y)
    {
        /* Bilinear Interpolation ( find xy with four points )
        (x1,y2)   (x2,y2)
            (x,y)
        (x1,y1)   (x2,y1)
        p(x,y) ~ {(y2-y)/(y2-y1)}*p(x,y1) + {(y-y1)/(y2-y1)}*p(x,y2)
               ~ {(y2-y)/(y2-y1)}*[{(x2-x)/(x2-x1)}*p(x1,y1) + {(x-x1)/(x2-x1)}*p(x2,y1)]
                 + {(y-y1)/(y2-y1)}*[{(x2-x)/(x2-x1)}*p(x1,y2) + {(x-x1)/(x2-x1)}*p(x2,y2)]
               = {1/(x2-x1)(y2-y1)}*[{p(x1,y1)(x2-x)(y2-y)}+{p(x2,y1)(x-x1)(y2-y)}
                                     +{p(x1,y2)(x2-x)(y-y1)}+{p(x2,y2)(x-x1)(y-y1)}]
        */
        double x1 = floor(x);
        double y1 = floor(y);
        double x2 = ceil(x);
        double y2 = ceil(y);
        vector<double> p11, p21, p12, p22;
        for (int i = 0; i < 3; i++)
        {
            p11.push_back(imgR.at<Vec3b>(y1, x1)[i]);
            p21.push_back(imgR.at<Vec3b>(y1, x2)[i]);
            p12.push_back(imgR.at<Vec3b>(y2, x1)[i]);
            p22.push_back(imgR.at<Vec3b>(y2, x2)[i]);
        }
        vector<double> p;
        /*
        p = {1/(x2-x1)(y2-y1)}*[{p(x1,y1)(x2-x)(y2-y)}+{p(x2,y1)(x-x1)(y2-y)}
                                     +{p(x1,y2)(x2-x)(y-y1)}+{p(x2,y2)(x-x1)(y-y1)}]
          = frac * [{pp11} + {pp21} + {pp12} + {pp22}]
        */
        for (int i = 0; i < 3; i++)
        {
            double frac = 1 / ((x2 - x1) * (y2 - y1));
            double pp11 = p11[i] * ((x2 - x) * (y2 - y));
            double pp21 = p21[i] * ((x - x1) * (y2 - y));
            double pp12 = p12[i] * ((x2 - x) * (y - y1));
            double pp22 = p22[i] * ((x - x1) * (y - y1));
            p.push_back(frac * (pp11 + pp21 + pp12 + pp22));
        }
        return p;
    }
};

#endif