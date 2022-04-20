#ifndef SerialMean_h
#define SerialMean_h

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class SerialMean
{
public:
	Mat srcImg;
	int width;
	int height;

	SerialMean(const Mat &_srcImg)
	{
		srcImg = _srcImg;
		width = srcImg.cols;
		height = srcImg.rows;
	}
	Mat myMean()
	{
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);

		uchar *srcData = (uchar *)srcImg.data;
		uchar *dstData = (uchar *)dstImg.data;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dstData[y * width + x] = Convolution(srcData, x, y);
			}
		}
		return dstImg;
	}

	int Convolution(uchar *arr, int x, int y)
	{
		int sum = 0;
		for (int j = -1; j <= 1; j++)
		{
			for (int i = -1; i <= 1; i++)
			{
				if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
				{
					sum += arr[(y + j) * width + (x + i)];
				}
			}
		}
		return sum / 9;
	}

	/*
	Reference : https://stackoverflow.com/a/67593119
	*/
	Mat optimizeMean()
	/*
		Take advantage of redundancy when summing consecutive columns.
		Save the last two column sums so that only one new column sum needs to be calculated in each iteration.
	*/
	{
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);

		uchar *srcData = (uchar *)srcImg.data;
		uchar *dstData = (uchar *)dstImg.data;

		for (int y = 0; y < height; y++)
		{
			int prev, curr, next;
			prev = curr = 0;
			for (int j = y - 1; j <= y + 1; j++)
			{
				prev += srcData[(j)*width + (0)];
				curr += srcData[(j)*width + (1)];
			}
			for (int x = 0; x < width; x++)
			{
				next = 0;
				for (int j = y - 1; j <= y + 1; j++)
				{
					next += srcData[(j)*width + (x + 1)];
				}
				dstData[y * width + x] = (prev + curr + next + 4) / 9;
				prev = curr;
				curr = next;
			}
		}
		return dstImg;
	}

	Mat moreOptimizeMean()
	/*
		As per optimizeMean, but also remove OpenCV overheads by caching pointers to each row.
					  | [0] | [1] | [2] |
		prev (y-1) -> | 0,0 | 0,1 | 0,2 |
		curr ( y ) -> | 1,0 | 1,1 | 1,2 |
		next (y+1) -> | 2,0 | 2,1 | 2,2 |
		If access to (1,1) then curr[1]
	*/
	{
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);

		// careful starring with 1
		for (int y = 1; y < height - 1; y++)
		{
			// const uint8_t에 대한 const 포인터, 주소와 값 모두 변경 불가능
			const uint8_t *const prev = srcImg.ptr(y - 1);
			const uint8_t *const curr = srcImg.ptr(y);
			const uint8_t *const next = srcImg.ptr(y + 1);
			uint8_t *const dst = dstImg.ptr(y);

			int _prev = prev[0] + curr[0] + next[0];
			int _curr = prev[1] + curr[1] + next[1];

			for (int x = 1; x < width - 1; x++)
			{
				int _next = prev[x + 1] + curr[x + 1] + next[x + 1];
				// 4 is for avoid 0
				dst[x] = (_prev + _curr + _next + 4) / 9;
				// update
				_prev = _curr;
				_curr = _next;
			}
		}
		return dstImg;
	}
};

#endif
