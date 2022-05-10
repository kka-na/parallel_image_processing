//* C++

#include <tchar.h>
#include <smmintrin.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <fstream>
#include <opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
void SSEmean_16bit(UINT16 * src, int height, int width, UINT16* output)
{
	__m128i  temp, temp1, temp2;
	__m128i  row00;
	__m128i  row01;
	__m128i  row02;

	__m128i  row10;
	__m128i  row11;
	__m128i  row12;

	__m128i  row20;
	__m128i  row21;
	__m128i  row22;

	__m128i cof1 = _mm_set1_epi16(0);
	__m128 ps1, ps2, ps3, ps4;
	__m128i WtoDW0_0, WtoDW0_1;
	__m128i w1, w2, wLast;
	__m128 nine = _mm_set_ps1(9.0f);


	int nNewwidth = width / 8;
	for (int y = 0; y < height - 2; y++) {

		__m128i* dstLast = (__m128i*)output + y*(nNewwidth)+nNewwidth;

		for (int x = 0; x < nNewwidth; x++) {
	//////////////////////////////////////////////////
			row00 = _mm_load_si128((__m128i*)src + x + y * nNewwidth);
			row10 = _mm_load_si128((__m128i*)src + x + (y + 1)*nNewwidth);
			row20 = _mm_load_si128((__m128i*)src + x + (y + 2)*nNewwidth);

			temp = _mm_load_si128((__m128i*)src + x + y * nNewwidth + 1);
			row01 = _mm_alignr_epi8(temp, row00, 2);
			row02 = _mm_alignr_epi8(temp, row00, 4);
			__m128i sumrow1 = _mm_add_epi16(_mm_add_epi16(row00, row01), row02);

			temp1 = _mm_load_si128((__m128i*)src + x + (y + 1) * nNewwidth + 1);
			row11 = _mm_alignr_epi8(temp1, row10, 2);
			row12 = _mm_alignr_epi8(temp1, row10, 4);
			__m128i sumrow2 = _mm_add_epi16(_mm_add_epi16(row10, row11), row12);

			temp2 = _mm_load_si128((__m128i*)src + x + (y + 2) * nNewwidth + 1);
			row21 = _mm_alignr_epi8(temp2, row20, 2);
			row22 = _mm_alignr_epi8(temp2, row20, 4);
			__m128i sumrow3 = _mm_add_epi16(_mm_add_epi16(row20, row21), row22);
			__m128i sum3X3 = _mm_add_epi16(_mm_add_epi16(sumrow1, sumrow2), sumrow3);

			WtoDW0_0 = _mm_unpacklo_epi16(sum3X3, cof1);
			WtoDW0_1 = _mm_unpackhi_epi16(sum3X3, cof1);
			ps1 = _mm_cvtepi32_ps(WtoDW0_0);
			ps2 = _mm_cvtepi32_ps(WtoDW0_1);

			ps1 = _mm_div_ps(ps1, nine);
			ps2 = _mm_div_ps(ps2, nine);

			WtoDW0_0 = _mm_cvtps_epi32(ps1);
			WtoDW0_1 = _mm_cvtps_epi32(ps2);
			
	/////////////////////////////////////////////

			// Pack 8 elements to 16 elements 
			wLast = _mm_packus_epi16(WtoDW0_0, WtoDW0_1);

			// store the result
			_mm_store_si128(dstLast + x, wLast);


		}
	}

 }

void SSEmean_8bit(unsigned char * src, int height, int width, unsigned char* output) 
{
}

int main()
{


	Mat src_image = imread("(Gray_512)baboon.jpg", 0);// load image using OpenCV
	resize(src_image, src_image, Size(4800, 4800));
	ushort* output = (ushort*)_mm_malloc(src_image.rows*src_image.cols * sizeof(ushort), 16);

	//for 16bit image
	uchar* output_8bit = (uchar*)_mm_malloc(src_image.rows*src_image.cols * sizeof(uchar), 8);
	Mat src_image_16;// = Mat::zeros(src_image.rows, src_image.cols, CV_16UC1);;
	src_image.convertTo(src_image_16, CV_16UC1);

	int width = src_image.cols;
	int height = src_image.rows;
	clock_t before, now;
	before = clock();

	for (int i = 0; i < 10; i++)
		SSEmean_8bit(src_image.data, src_image.rows, src_image.cols, output_8bit);

	now = clock();
	printf("Processing Time(8bit): %lf msec\n", (double)(now - before));


	before = clock();

	for (int i = 0; i < 10; i++)
		SSEmean_16bit((ushort*)src_image_16.data, src_image.rows, src_image.cols, output);

	now = clock();
	printf("Processing Time(16bit): %lf msec\n", (double)(now - before));


	Mat imgd1(Size(src_image.rows, src_image.cols), CV_8UC1, output_8bit);
	Mat imgd2(Size(src_image.rows, src_image.cols), CV_16UC1, output);
	Mat dstdiplay;
	normalize(imgd1, dstdiplay, 255, 0, NORM_MINMAX, CV_8UC1);
	imgd2.convertTo(imgd2, CV_8UC1);

	Mat diff = imgd1 - imgd2;
	cv::Scalar sum1 = sum(diff);
	cout << sum1 << endl;


	namedWindow("img_ori", 0);
	namedWindow("img_mean_8bit", 0);
	namedWindow("img_mean_16bit", 0);
	imshow("img_ori", src_image);
	imshow("img_mean_8bit", imgd1);
	imshow("img_mean_16bit", imgd2);
	waitKey(0);
	destroyAllWindows();

	return 0;
}
//*/