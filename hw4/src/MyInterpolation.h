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

    MyInterpolation(Mat _src)
    {
        src = _src;
        h = src.rows;
        w = src.cols;
    }
    Mat SerialInterpolation(int type, int npixel)
    {
        int hWp = h * npixel;
        int wWp = w * npixel;
        Mat result = Mat::zeros(hWp, wWp, src.type());
        resize(src, dst, Size(hWp, wWp), 0, 0, INTER_NEAREST);
        for (int r = 0; r < hWp; r++)
        {
            for (int c = 0; c < wWp; c++)
            {
                double x = c;
                double y = r;
                if ((x < dst.cols - 1) && (y < dst.rows - 2) && (x > 2) && (y > 2))
                {
                    vector<double> itpRGB;
                    switch (type)
                    {
                    case 1:
                        itpRGB = Bilinear(x, y);
                        break;
                    case 2:
                        itpRGB = BiCubic(x, y);
                        break;
                    default:
                        itpRGB = Bilinear(x, y);
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
        int hWp = h * npixel;
        int wWp = w * npixel;
        Mat result = Mat::zeros(hWp, wWp, src.type());
        resize(src, dst, Size(hWp, wWp));
        int r, c;
#pragma omp parallel for private(r, c) ordered
        for (r = 0; r < hWp; r++)
        {
            for (c = 0; c < wWp; c++)
            {
                double x = c;
                double y = r;
                if ((x < dst.cols - 2) && (y < dst.rows - 2) && (x > 2) && (y > 2))
                {
                    vector<double> itpRGB;
                    switch (type)
                    {
                    case 1:
                        itpRGB = Bilinear(x, y);
                        break;
                    case 2:
                        itpRGB = BiCubic(x, y);
                        break;
                    default:
                        itpRGB = Bilinear(x, y);
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
        double x1 = x - 1;
        double y1 = y + 1;
        double x2 = x + 1;
        double y2 = y - 1;
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
            double frac = 1 / ((x2 - x1) * (y2 - y1));
            double pp11 = ((x2 - x) * (y2 - y)) != 0 ? p11[i] * ((x2 - x) * (y2 - y)) : p11[i];
            double pp21 = ((x - x1) * (y2 - y)) != 0 ? p21[i] * ((x - x1) * (y2 - y)) : p21[i];
            double pp12 = ((x2 - x) * (y - y1)) != 0 ? p12[i] * ((x2 - x) * (y - y1)) : p12[i];
            double pp22 = ((x - x1) * (y - y1)) != 0 ? p22[i] * ((x - x1) * (y - y1)) : p22[i];
            p.push_back(frac * (pp11 + pp21 + pp12 + pp22));
        }
        return p;
    }
    vector<double> BiCubic(double x, double y)
    {
        /*
         p = p00 + p10x + p20x^2 + p30x^3 + p01y + p02y^2 + p03y^3 +
            p11xy + p21x^2y + p31x^3y + p12xy^2 + p22x^2y^2 + p32xrac * (pp11 + pp21 + pp12 + pp22)^3y^2 +
            p13xy^3 + p23x^2y^3 + p33x^3y^3
        */
        double x0 = x - 2;
        double y0 = y - 2;
        double x1 = x - 1;
        double y1 = y - 1;
        double x2 = x + 1;
        double y2 = y + 1;
        double x3 = x + 2;
        double y3 = y + 2;

        vector<double> p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33;
        for (int i = 0; i < 3; i++)
        {
            p00.push_back(dst.at<Vec3b>(y0, x0)[i]);
            p01.push_back(dst.at<Vec3b>(y1, x0)[i]);
            p02.push_back(dst.at<Vec3b>(y2, x0)[i]);
            p03.push_back(dst.at<Vec3b>(y3, x0)[i]);
            p10.push_back(dst.at<Vec3b>(y0, x1)[i]);
            p11.push_back(dst.at<Vec3b>(y1, x1)[i]);
            p12.push_back(dst.at<Vec3b>(y2, x1)[i]);
            p13.push_back(dst.at<Vec3b>(y3, x1)[i]);
            p20.push_back(dst.at<Vec3b>(y0, x2)[i]);
            p21.push_back(dst.at<Vec3b>(y1, x2)[i]);
            p22.push_back(dst.at<Vec3b>(y2, x2)[i]);
            p23.push_back(dst.at<Vec3b>(y3, x2)[i]);
            p30.push_back(dst.at<Vec3b>(y0, x3)[i]);
            p31.push_back(dst.at<Vec3b>(y1, x3)[i]);
            p32.push_back(dst.at<Vec3b>(y2, x3)[i]);
            p33.push_back(dst.at<Vec3b>(y3, x3)[i]);
        }
        vector<double> p;
        for (int i = 0; i < 3; i++)
        {
            double a00 = p00[i];
            double a01 = -double(1 / 2) * p10[i] + double(1 / 2) * p12[i];
            double a02 = p10[i] - double(5 / 2) * p11[i] + 2 * p12[i] - double(1 / 2) * p13[i];
            double a03 = -double(1 / 2) * p10[i] + double(3 / 2) * p11[i] - double(3 / 2) * p12[i] + double(1 / 2) * p13[i];
            double a10 = -double(1 / 2) * p01[i] + double(1 / 2) * p21[i];
            double a11 = double(1 / 4) * p00[i] - double(1 / 4) * p02[i] - double(1 / 4) * p20[i] + double(1 / 4) * p22[i];
            double a12 = -double(1 / 2) * p00[i] + double(5 / 4) * p01[i] - p02[i] + double(1 / 4) * p00[i] + double(1 / 2) * p20[i] - double(5.4) * p21[i] + p22[i] - double(1 / 4) * p23[i];
            double a13 = double(1 / 4) * p00[i] - double(3 / 4) * p01[i] + double(3 / 4) * p02[i] - double(1 / 4) * p03[i] - double(1 / 4) * p20[i] + double(3 / 4) * p21[i] - double(3 / 4) * p22[i] + double(1 / 4) * p23[i];
            double a20 = p00[i] - double(5 / 2) * p11[i] + 2 * p21[i] - double(1 / 2) * p31[i];
            double a21 = -double(1 / 2) * p00[i] + double(1 / 2) * p02[i] + double(5 / 4) * p10[i] - double(5 / 4) * p12[i] - p20[i] + p22[i] + double(1 / 4) * p30[i] - double(1 / 4) * p32[i];
            double a22 = p00[i] - double(5 / 2) * p01[i] + 2 * p02[i] - double(1 / 2) * p03[i] - double(5 / 2) * p10[i] + double(25 / 4) * p11[i] - 5 * p12[i] + double(5 / 4) * p13[i] + 2 * p20[i] - 5 * p21[i] + 4 * p22[i] - p23[i] - double(1 / 2) * p30[i] + double(5 / 4) * p31[i] - p32[i] + double(1 / 4) * p33[i];
            double a23 = -double(1 / 2) * p00[i] + double(3 / 2) * p01[i] - double(3 / 2) * p02[i] + double(1 / 2) * p03[i] + double(5 / 4) * p10[i] - double(15 / 4) * p11[i] + double(15 / 4) * p12[i] - double(5 / 4) * p13[i] - p20[i] + 3 * p21[i] - 3 * p22[i] + p23[i] + double(1 / 4) * p30[i] - double(3 / 4) * p31[i] + double(3 / 4) * p32[i] - double(1 / 4) * p33[i];
            double a30 = -double(1 / 2) * p00[i] + double(3 / 2) * p11[i] - double(3 / 2) * p21[i] + double(1 / 2) * p31[i];
            double a31 = double(1 / 4) * p00[i] - double(1 / 4) * p02[i] - double(3 / 4) * p10[i] + double(3 / 4) * p12[i] + double(3 / 4) * p20[i] - double(3 / 4) * p22[i] - double(1 / 4) * p30[i] + double(1 / 4) * p32[i];
            double a32 = -double(1 / 2) * p00[i] + double(5 / 4) * p01[i] - p02[i] + double(1 / 4) * p03[i] + double(3 / 2) * p10[i] - double(15 / 4) * p11[i] + 3 * p12[i] - double(3 / 4) * p13[i] - double(3 / 2) * p20[i] + double(15 / 4) * p21[i] - 3 * p22[i] + double(3 / 4) * p23[i] + double(1 / 2) * p30[i] - double(5 / 4) * p31[i] + p32[i] - double(1 / 4) * p33[i];
            ;
            double a33 = double(1 / 4) * p00[i] - double(3 / 4) * p01[i] + double(3 / 4) * p02[i] - double(1 / 4) * p03[i] - double(3 / 4) * p10[i] + double(9 / 4) * p11[i] - double(9 / 4) * p12[i] + double(3 / 4) * p13[i] + double(3 / 4) * p20[i] - double(9 / 4) * p21[i] + double(9 / 4) * p22[i] - double(3 / 4) * p23[i] - double(1 / 4) * p30[i] + double(3 / 4) * p31[i] - double(3 / 4) * p32[i] + double(1 / 4) * p33[i];
            ;

            p.push_back(a00 + (a01 + (a02 + a03 * y) * y) * y + (a10 + (a11 + (a12 + a13 * y) * y) * y) * x + (a20 + (a21 + (a22 + a23 * y) * y) * y) * (x * x) + (a30 + (a31 + (a32 + a33 * y) * y) * y) * (x * x * x));
        }
        return p;
    }
};

#endif