#ifndef MyInterpolation_h
#define MyInterpolation_h

#include <omp.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

class MyInterpolation
{
public:
    Mat src;
    int h, w;

    MyInterpolation(Mat _src){
        src = _src;
        h = src.rows;
        w = src.cols;
    }
    Mat Serial_Interpolation(int type, int npixel)
    {
        int hWp = h*npixel;
        int wWp = w*npixel;
        Mat result = Mat::zeros(hWp, wWp, src.type());    
        
        for (int r = 0; r < hWp; r++)
        {
            for (int c = 0; c < wWp; c++)
            {
                // using Bilinear Interpolation , can get
                double x = double(double(c)/double(npixel)-(int)(c/npixel));
                double y =double(double(r)/double(npixel)-(int)(r/npixel));
                if ((x < src.cols - 1) && (y < src.rows - 1) && (x > 1) && (y > 1))
                {
                    vector<double> itpRGB;
                    switch(type)
                    {
                    case 1:
                        itpRGB = Bilinear(x,y);
                        break;
                    
                    default:
                        itpRGB = Bilinear(x,y);
                        break;
                    }
                    for (int i = 0; i < 3; i++)
                    {
                        result.at<Vec3b>(r, c)[i] = itpRGB[i];
                    }
                }
            }
        }
        return result;
    }
    vector<double> Bilinear(double x, double y)
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
            p11.push_back(src.at<Vec3b>(y1, x1)[i]);
            p21.push_back(src.at<Vec3b>(y1, x2)[i]);
            p12.push_back(src.at<Vec3b>(y2, x1)[i]);
            p22.push_back(src.at<Vec3b>(y2, x2)[i]);
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