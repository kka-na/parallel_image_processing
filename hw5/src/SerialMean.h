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

	SerialMean(const Mat& _srcImg){
		srcImg = _srcImg;
		width = srcImg.cols; 
		height = srcImg.rows;
	}
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

	Mat scalarMean(){
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);

		uchar* srcData = (uchar*)srcImg.data;
		uchar* dstData = (uchar*)dstImg.data;

		for(int y = 0; y<height; y++){
			int prev,curr, next;
			prev=curr=0;
			for(int j = y-1; j<=y+1; j++){
				prev += srcData[(j)*width+(0)];
				curr += srcData[(j)*width+(1)];
			}
			for(int x=0; x<width; x++){
				next =0;
				for(int j = y-1; j<=y+1; j++){
					next += srcData[(j)*width+(x+1)];
				}
				dstData[y * width +x] = (prev+curr+next+4)/9;
				prev = curr;
				curr = next;
			}
		}
		return dstImg;
	}

};

#endif
