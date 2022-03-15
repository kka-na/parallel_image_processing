#ifndef SaltandPepper_h
#define SaltandPepper_h

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class SaltandPepper
{
public:
    Mat SpreadSaltsPepper(Mat img, int num){ // num의 수만큼 랜덤한 위치에 점 찍기 
        for(int n=0; n<num; n++){
            int x = rand()%img.cols;
            int y = rand()%img.rows;
            if (img.channels() == 1){
                int i = rand()%2;
                if(i == 0) img.at<uchar>(y,x) = 0;
                else img.at<uchar>(y,x) = 255;
            }else{
                int i = rand()%2;
                img.at<Vec3b>(y,x)[i] = 255;
                if(i == 0){
                    img.at<Vec3b>(y,x)[0] = 0;
                    img.at<Vec3b>(y,x)[1] = 0;
                    img.at<Vec3b>(y,x)[2] = 0;
                }else{
                    img.at<Vec3b>(y,x)[0] = 255;
                    img.at<Vec3b>(y,x)[1] = 255;
                    img.at<Vec3b>(y,x)[2] = 255;
                }
            }
        }
        return img;
    }
    void myMedian(const Mat& srcImg, Mat& dstImg, const Size& kn_size){
		dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
		int wd = srcImg.cols; int hg = srcImg.rows;
		int kwd = kn_size.width; int khg = kn_size.height;
		int rad_w = kwd / 2; int rad_h = khg / 2;
		uchar* srcData = (uchar*)srcImg.data;
		uchar* dstData = (uchar*)dstImg.data;

		vector<float> table;

		for(int c = rad_w + 1; c < wd - rad_w; c++){
			for(int r = rad_h + 1; r < hg - rad_h; r++){
				for( int kc = -rad_w; kc <= rad_w; kc++){
					for(int kr = - rad_h; kr<= rad_h; kr++){
						//sorting 위한 테이블에 커널안에 해당하는 값들을 벡터 테이블에 저장
						//픽셀에 mat 클래스 data에 접근하는 접근법 사용한다. 
						//.data는 데이터를 일렬로 되어있고, 해당 데이터에 포인터로 접근할 수 있다. 
						table.push_back((float)srcData[(r+kr)*wd+(c+kc)]);
					}
				}
				//start, end값을 이용하여 범위안의 인자를 오름차순으로 정렬한다. 
				sort(table.begin(), table.end());
				dstData[r * wd +c] = (uchar)table[(kwd*khg)/2]; //원소들의 수 / 2 한 값에 반올림 하여 중간값을 얻을 수 있다. 
				table.clear(); // 다음 커널로 옮겨가서 수행하므로 clear하여 벡터저장공간을 비워준다. 
			}
		}
	}
};

#endif /*SaltandPepper_h*/