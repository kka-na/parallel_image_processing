#ifndef SerialInterpolation_h
#define SerialInterpolation_h

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

class SerialInterpolation
{
private:
    Mat src;
    int h = 2448;
    int w = 3264;
    FILE *file;
    unsigned char *raw;
    unsigned short *data;
    unsigned short **r;
    unsigned short **g;
    unsigned short **b;

public:
    SerialInterpolation()
    {
        LoadData();
    }
    Mat doBayer()
    {
        Mat dst = Bayer();
        delMem();
        return dst;
    }

private:
    void LoadData()
    {
        file = fopen("../image/raw.RAW", "rb");
        fseek(file, 0, SEEK_END);
        long fsize = ftell(file);
        rewind(file);

        raw = new unsigned char[fsize];
        fread(raw, sizeof(char), w * h, file);
        data = new unsigned short[h * w];
        seq_data_copy(); // 8bit to 10 bit data
    }

    void seq_data_copy()
    {
        // unsigned short temp;
        for (int i = 0, j = 0; i < (h * w); i += 4, j += 5)
        {
            // for 10 bit
            data[i] = (raw[j] << 2) + ((raw[j + 4] >> 0) & 3);
            data[i + 1] = (raw[j + 1] << 2) + ((raw[j + 4] >> 2) & 3);
            data[i + 2] = (raw[j + 2] << 2) + ((raw[j + 4] >> 4) & 3);
            data[i + 3] = (raw[j + 3] << 2) + ((raw[j + 4] >> 6) & 3);
        }
    }
    Mat Bayer()
    {
        // Initialization
        r = new unsigned short *[w];
        g = new unsigned short *[w];
        b = new unsigned short *[w];
        for (int i = 0; i < w; i++)
        {
            r[i] = new unsigned short[w];
            g[i] = new unsigned short[w];
            b[i] = new unsigned short[w];
        }
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                if ((i % 2) == 0 && (j % 2) == 0)
                {
                    r[j][i] = data[j * w + i];
                }
                if ((i % 2) == 1 && (j % 2) == 0)
                {
                    g[j][i] = data[j * w + i];
                }
                if ((i % 2) == 0 && (j % 2) == 1)
                {
                    g[j][i] = data[j * w + i];
                }
                if ((i % 2) == 1 && (j % 2) == 1)
                {
                    b[j][i] = data[j * w + i];
                }
            }
        }

        RedInterpolation();
        GreenInterpolation();
        BlueInterpolation();

        unsigned short *r_interp = new unsigned short[h * w];
        unsigned short *g_interp = new unsigned short[h * w];
        unsigned short *b_interp = new unsigned short[h * w];

        int cnt = 0;
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                r_interp[j * w + i] = b[j][i];
                g_interp[j * w + i] = g[j][i];
                b_interp[j * w + i] = b[j][i];
            }
        }
        src = Change2Mat(data, data, data);
        Mat dst = Change2Mat(r_interp, g_interp, b_interp);
        return dst;
    }

    void RedInterpolation()
    {
        for (int j = 0; j < h; j += 2)
        {
            for (int i = 0; i < w; i += 2)
            {
                if (i < w - 2 && j < h - 2)
                {
                    r[j][i + 1] = (r[j][i] + r[j][i + 2]) / 2;
                    r[j + 1][i + 1] = (r[j][i] + r[j][i + 2] + r[j + 2][i] + r[j + 2][i + 2]) / 4;
                }
            }
        }
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j += 2)
            {
                if (j < h - 2)
                {
                    r[j + 1][i] = (r[j][i] + r[j + 2][i]) / 2;
                }
            }
        }
    }

    void BlueInterpolation()
    {
        for (int j = 1; j < h; j++)
        {
            for (int i = 1; i < w; i += 2)
            {
                if (i < w - 2)
                {
                    b[j][i + 1] = (b[j][i] + b[j][i + 2]) / 2;
                    b[j + 1][i + 1] = (b[j][i] + b[j][i + 2] + b[j + 2][i] + b[j + 2][i + 2]) / 4;
                }
            }
        }
        for (int i = 1; i < w; i++)
        {
            for (int j = 1; j < h; j += 2)
            {
                if (j < h - 2)
                {
                    b[j + 1][i] = (b[j][i] + b[j + 2][i]) / 2;
                }
                b[j][0] = b[j][1];
            }
            b[0][i] = b[1][i];
        }
    }

    void GreenInterpolation()
    {
        for (int j = 1; j < h; j += 2)
        {
            for (int i = 1; i < w; i += 2)
            {
                if (i < w - 2 && j < h - 2)
                {
                    g[j][i] = (g[j - 1][i] + g[j][i - 1] + g[j + 1][i] + g[j][i + 1]) / 4;
                }
            }
        }
        for (int i = 2; i < w; i += 2)
        {
            for (int j = 2; j < h; j += 2)
            {
                if (i < w - 1 && j < h - 1)
                {
                    g[j][i] = (g[j - 1][i] + g[j][i - 1] + g[j + 1][i] + g[j][i + 1]) / 4;
                }
            }
        }
        for (int i = 0; i < w; i += 2)
        {
            if (i > 0)
            {
                g[0][i] = (g[0][i - 1] + g[0][i + 1] + g[1][i]) / 3;
            }
            else if (i == 0)
            {
                g[0][i] = (g[0][i + 1] + g[1][i]) / 2;
            }
        }
        for (int i = 1; i < w; i += 2)
        {
            if (i < w - 1)
            {
                g[h - 1][i] = (g[h - 1][i - 1] + g[h - 1][i + 1] + g[h - 2][i]) / 3;
            }
            else if (i == w - 1)
            {
                g[h - 1][i] = (g[h - 1][i - 1] + g[h - 2][i]) / 2;
            }
        }
        for (int j = 2; j < h; j += 2)
        {
            g[j][0] = (g[j - 1][0] + g[j][1] + g[j + 1][0]) / 3;
        }
        for (int j = 1; j < h - 1; j += 2)
        {
            g[j][w - 1] = (g[j - 1][w - 1] + g[j][w - 2] + g[j + 1][w + 1]) / 3;
        }
    }

    Mat Change2Mat(unsigned short *r_data, unsigned short *g_data, unsigned short *b_data)
    {
        Mat r_mat161(h, w, CV_16UC1);
        Mat g_mat161(h, w, CV_16UC1);
        Mat b_mat161(h, w, CV_16UC1);
        int cnt = 0;
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                r_mat161.at<short>(j, i) = b_data[j * w + i];
                g_mat161.at<short>(j, i) = g_data[j * w + i];
                b_mat161.at<short>(j, i) = r_data[j * w + i];
            }
        }

        Mat mat163(h, w, CV_16UC3);
        vector<Mat> vec;
        vec.push_back(r_mat161);
        vec.push_back(g_mat161);
        vec.push_back(b_mat161);
        merge(vec, mat163);

        normalize(mat163, mat163, 0, 255, NORM_MINMAX);
        // SaveRawFile10bit((unsigned short *)mat163.data);
        Mat mat83(h, w, CV_8UC3);
        mat163.convertTo(mat83, CV_8UC3);
        return mat83;
    }

    void SaveRawFile10bit(unsigned short *data)
    {
        FILE *fp;
        fp = fopen("../result/serial.raw", "wb");
        fwrite(data, sizeof(unsigned short), h * w * 3, fp);
        fclose(fp);
    }
    void delMem()
    {
        for (int i = 0; i < w; i++)
        {
            delete[] r[i];
            delete[] g[i];
            delete[] b[i];
        }
        delete[] r;
        delete[] g;
        delete[] b;

        delete[] raw;
        delete[] data;

        fclose(file);
    }
};

#endif