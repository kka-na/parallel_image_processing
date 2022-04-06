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
    Mat dst;
    int h, w;

    MyInterpolation(Mat _src){
        src = _src;
        h = src.rows;
        w = src.cols;
    }
    Mat SerialInterpolation(int type, int npixel)
    {
        int hWp = h*npixel;
        int wWp = w*npixel;
        Mat result  = Mat::zeros(hWp, wWp, src.type());
        resize(src, dst, Size(hWp, wWp));
        
        for (int r = 0; r < hWp; r++)
        {
            for (int c = 0; c < wWp; c++)
            {
                if ((c < dst.cols - 1) && (r < dst.rows - 1) && (c > 1) && (r > 1))
                {
                    vector<double> itpRGB;
                    switch(type)
                    {
                    case 1:
                        itpRGB = Bilinear(c,r);
                        break;
                    
                    default:
                        itpRGB = Bilinear(c,r);
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
    Mat OMPInterpolation(int type, int npixel)
    {
        int hWp = h*npixel;
        int wWp = w*npixel;
        Mat result  = Mat::zeros(hWp, wWp, src.type());
        resize(src, dst, Size(hWp, wWp));
        int r,c;
#pragma omp parallel for private(r) ordered
        for (r = 0; r < hWp; r++)
        {
            #pragma omp ordered
            for (c = 0; c < wWp; c++)
            {
                if ((c < dst.cols - 1) && (r < dst.rows - 1) && (c > 1) && (r > 1))
                {
                    vector<double> itpRGB;
                    switch(type)
                    {
                    case 1:
                        itpRGB = Bilinear(c,r);
                        break;
                    
                    default:
                        itpRGB = Bilinear(c,r);
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
        double x1 = x-1;
        double y1 = y+1;
        double x2 = x+1;
        double y2 = y-1;
        vector<double> p11, p21, p12, p22;
        for (int i = 0; i < 3; i++)
        {
            p11.push_back(dst.at<Vec3b>(y1, x1)[i]);
            p21.push_back(dst.at<Vec3b>(y1, x2)[i]);
            p12.push_back(dst.at<Vec3b>(y2, x1)[i]);
            p22.push_back(dst.at<Vec3b>(y2, x2)[i]);
        }
        vector<double> p;
        /*
        p = {1/(x2-x1)(y2-y1)}*[{p(x1,y1)(x2-x)(y2-y)}+{p(x2,y1)(x-x1)(y2-y)}
                                     +{p(x1,y2)(x2-x)(y-y1)}+{p(x2,y2)(x-x1)(y-y1)}]
          = frac * [{pp11} + {pp21} + {pp12} + {pp22}]
        */
        for (int i = 0; i < 3; i++)
        {
            double frac =1 / ((x2 - x1) * (y2 - y1));//((x2 - x1) * (y2 - y1)) != 0 ? 1 / ((x2 - x1) * (y2 - y1)) : 1;
            double pp11 =((x2 - x) * (y2 - y)) != 0  ? p11[i] * ((x2 - x) * (y2 - y)) : p11[i];
            double pp21 = ((x - x1) * (y2 - y)) != 0 ? p21[i] * ((x - x1) * (y2 - y)) : p21[i];
            double pp12 = ((x2 - x) * (y - y1)) !=0 ? p12[i] * ((x2 - x) * (y - y1)) : p12[i];
            double pp22 = ((x2 - x) * (y - y1)) !=0 ? p22[i] * ((x - x1) * (y - y1)) : p22[i];
            p.push_back(frac * (pp11 + pp21 + pp12 + pp22));
        }
        return p;
    }
};

#endif