#ifndef SSEMean_h
#define SSEMean_h

#include <iostream>
#include <tmmintrin.h>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

class SSEMean
{
	public:

	Mat srcImg;
	int width;
	int height;

	SSEMean(const Mat& _srcImg){
		srcImg = _srcImg;
		width = srcImg.cols; 
		height = srcImg.rows;
	}
	//https://stackoverflow.com/questions/67547630/sse-mean-filter-in-c-and-opencv
    Mat myMean(){
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);

		uchar* srcData = (uchar*)srcImg.data;
		uchar* dstData = (uchar*)dstImg.data;

		for(int y = 0; y<height; y++){
			for(int x=0; x<width; x++){
				dstData[y * width +x] = Convolution(srcData, x, y);
			}
		}
		return dstImg;
	}

	int Convolution(uchar *arr, int x, int y){
		int sum = 0;
		for(int j = -1; j<=1; j++){
			for(int i=-1; i<=1; i++){
				if( (y+j)>=0 && (y+j)<height && (x+i)>=0 && (x+i)<width){
					sum += arr[(y+j)*width+(x+i)];
				}
			}
		}
		return sum/9;
	}
};

#endif
