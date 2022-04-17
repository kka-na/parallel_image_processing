#include <iostream>
#include <tchar.h>
#include <smmintrin.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;
Mat approach_8bit(Mat img, string mode);
Mat approach_16bit(Mat img, string mode);

//select 
string mode = "16";  // "8bit or 16bit" 

void main() {
	Mat img = imread("lena.jpg", 0);
	Mat sse_result(img.size().height, img.size().width, CV_8UC1);
	int64 tend, tStart;

	string title = "Load Image";
	namedWindow(title,0);
	imshow(title, img);

	if (mode == "8") {
		cout << "Using 8bit approach" << endl;
		tStart = getTickCount();
		sse_result = approach_8bit(img, mode);
		tend = getTickCount() - tStart;
	}
	if (mode == "16") {
		cout << "Using 16bit approach" << endl;
		tStart = getTickCount();
		sse_result = approach_16bit(img, mode);
		tend = getTickCount() - tStart;
		
	}

	cout << mode << "bit Processing time of SSE :" << tend / getTickFrequency() << "ms" << endl;
	string outImg = "Outpit Image";
	namedWindow(outImg, 0);
	imshow(outImg, sse_result);
	waitKey(0);
}

Mat approach_8bit(Mat img, string mode) {

	uchar* src = img.data;

	int width = img.size().width;
	int height = img.size().height;

	Mat sse_result(height, width, CV_8UC1);
	uchar* dst = sse_result.data;

	__m128i* temp0, *temp1, *temp2, *row00, *row10, *row20;
	__m128i row01, row02, row11, row12, row21, row22;
	__m128i row00_l, row00_h, row01_l, row01_h, row02_l, row02_h;
	__m128i row10_l, row10_h, row11_l, row11_h, row12_l, row12_h;
	__m128i row20_l, row20_h, row21_l, row21_h, row22_l, row22_h;
	__m128i result, result_l, reulst_h, Vsum_l, Vsum_h, sumrow0_l, sumrow0_h, sumrow1_l, sumrow1_h, sumrow2_l, sumrow2_h;
	__m128i	unpack_l_l, unpack_l_h, unpack_h_l, unpack_h_h, unpack_l_l2int, unpack_l_h2int, unpack_h_l2int, unpack_h_h2int;
	__m128i m0 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
	__m128  cvt_unpack_l_l, cvt_unpack_l_h, cvt_unpack_h_l, cvt_unpack_h_h, div_l_l, div_l_h, div_h_l, div_h_h;
	__m128 div_9 = _mm_set_ps1(9.0);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width / 16; x++) {
			row00 = (__m128i*)src + x + y * (width / 16);
			row00_l = _mm_unpacklo_epi8(*row00, m0);
			row00_h = _mm_unpackhi_epi8(*row00, m0);

			temp0 = (__m128i*)src + x + y * (width / 16) + 1;

			row01 = _mm_alignr_epi8(*temp0, *row00, 1);
			row01_l = _mm_unpacklo_epi8(row01, m0);
			row01_h = _mm_unpackhi_epi8(row01, m0);

			row02 = _mm_alignr_epi8(*temp0, *row00, 2);
			row02_l = _mm_unpacklo_epi8(row02, m0);
			row02_h = _mm_unpackhi_epi8(row02, m0);

			sumrow0_l = _mm_add_epi16(_mm_add_epi16(row00_l, row01_l), row02_l);
			sumrow0_h = _mm_add_epi16(_mm_add_epi16(row00_h, row01_h), row02_h);

			row10 = (__m128i*)src + x + (y + 1) * (width / 16);
			row10_l = _mm_unpacklo_epi8(*row10, m0);
			row10_h = _mm_unpackhi_epi8(*row10, m0);

			temp1 = (__m128i*)src + x + (y + 1) * (width / 16) + 1;

			row11 = _mm_alignr_epi8(*temp1, *row10, 1);
			row11_l = _mm_unpacklo_epi8(row11, m0);
			row11_h = _mm_unpackhi_epi8(row11, m0);

			row12 = _mm_alignr_epi8(*temp1, *row10, 2);
			row12_l = _mm_unpacklo_epi8(row12, m0);
			row12_h = _mm_unpackhi_epi8(row12, m0);

			sumrow1_l = _mm_add_epi16(_mm_add_epi16(row10_l, row11_l), row12_l);
			sumrow1_h = _mm_add_epi16(_mm_add_epi16(row10_h, row11_h), row12_h);

			row20 = (__m128i*)src + x + (y + 2) * (width / 16);
			row20_l = _mm_unpacklo_epi8(*row20, m0);
			row20_h = _mm_unpackhi_epi8(*row20, m0);

			temp2 = (__m128i*)src + x + (y + 2) * (width / 16) + 1;

			row21 = _mm_alignr_epi8(*temp2, *row20, 1);
			row21_l = _mm_unpacklo_epi8(row21, m0);
			row21_h = _mm_unpackhi_epi8(row21, m0);

			row22 = _mm_alignr_epi8(*temp2, *row20, 2);
			row22_l = _mm_unpacklo_epi8(row22, m0);
			row22_h = _mm_unpackhi_epi8(row22, m0);

			sumrow2_l = _mm_add_epi16(_mm_add_epi16(row20_l, row21_l), row22_l);
			sumrow2_h = _mm_add_epi16(_mm_add_epi16(row20_h, row21_h), row22_h);

			Vsum_l = _mm_add_epi16(_mm_add_epi16(sumrow0_l, sumrow1_l), sumrow2_l);
			Vsum_h = _mm_add_epi16(_mm_add_epi16(sumrow0_h, sumrow1_h), sumrow2_h);

			unpack_l_l = _mm_unpacklo_epi16(Vsum_l, m0);
			unpack_l_h = _mm_unpackhi_epi16(Vsum_l, m0);

			unpack_h_l = _mm_unpacklo_epi16(Vsum_h, m0);
			unpack_h_h = _mm_unpackhi_epi16(Vsum_h, m0);

			cvt_unpack_l_l = _mm_cvtepi32_ps(unpack_l_l);
			cvt_unpack_l_h = _mm_cvtepi32_ps(unpack_l_h);

			cvt_unpack_h_l = _mm_cvtepi32_ps(unpack_h_l);
			cvt_unpack_h_h = _mm_cvtepi32_ps(unpack_h_h);

			div_l_l = _mm_div_ps(cvt_unpack_l_l, div_9);
			div_l_h = _mm_div_ps(cvt_unpack_l_h, div_9);

			div_h_l = _mm_div_ps(cvt_unpack_h_l, div_9);
			div_h_h = _mm_div_ps(cvt_unpack_h_h, div_9);

			unpack_l_l = _mm_cvtps_epi32(div_l_l);
			unpack_l_h = _mm_cvtps_epi32(div_l_h);

			unpack_h_l = _mm_cvtps_epi32(div_h_l);
			unpack_h_h = _mm_cvtps_epi32(div_h_h);

			result_l = _mm_packs_epi32(unpack_l_l, unpack_l_h);
			reulst_h = _mm_packs_epi32(unpack_h_l, unpack_h_h);

			result = _mm_packus_epi16(result_l, reulst_h);

			_mm_store_si128((__m128i*)dst + x + y * (width / 16), result);
		}
	}
	return sse_result;
}

Mat approach_16bit(Mat img, string mode) {

	int width = img.size().width;
	int height = img.size().height;

	img.convertTo(img, CV_16UC1);
	uchar * src = img.data;

	Mat sse_result(height, width, CV_8UC1);
	sse_result.convertTo(sse_result, CV_16UC1);
	uchar * dst = sse_result.data;

	__m128i *temp0, *temp1, *temp2, *row00, *row10, *row20;
	__m128i row01, row02, row11, row12, row21, row22;
	__m128i result, Vsum, sumrow0, sumrow1, sumrow2, unpack_l, unpack_h;
	__m128i m0 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
	__m128  cvt_unpack_l, cvt_unpack_h, div_l, div_h;
	__m128 div_9 = _mm_set_ps1(9.0);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width / 8; x++) {
			row00 = (__m128i*)src + x + y * (width / 8);
			row10 = (__m128i*)src + x + (y + 1)*(width / 8);
			row20 = (__m128i*)src + x + (y + 2)*(width / 8);

			temp0 = (__m128i*)src + x + y * (width / 8) + 1;
			temp1 = (__m128i*)src + x + (y + 1) * (width / 8) + 1;
			temp2 = (__m128i*)src + x + (y + 2) * (width / 8) + 1;

			row01 = _mm_alignr_epi8(*temp0, *row00, 2);
			row02 = _mm_alignr_epi8(*temp0, *row00, 4);
			sumrow0 = _mm_add_epi16(_mm_add_epi16(*row00, row01), row02);

			row11 = _mm_alignr_epi8(*temp1, *row10, 2);
			row12 = _mm_alignr_epi8(*temp1, *row10, 4);
			sumrow1 = _mm_add_epi16(_mm_add_epi16(*row10, row01), row02);

			row21 = _mm_alignr_epi8(*temp2, *row20, 2);
			row22 = _mm_alignr_epi8(*temp2, *row20, 4);
			sumrow2 = _mm_add_epi16(_mm_add_epi16(*row20, row01), row02);

			Vsum = _mm_add_epi16(_mm_add_epi16(sumrow0, sumrow1), sumrow2);

			unpack_l = _mm_unpacklo_epi16(Vsum, m0);
			unpack_h = _mm_unpackhi_epi16(Vsum, m0);

			cvt_unpack_l = _mm_cvtepi32_ps(unpack_l);
			cvt_unpack_h = _mm_cvtepi32_ps(unpack_h);

			div_l = _mm_div_ps(cvt_unpack_l, div_9);
			div_h = _mm_div_ps(cvt_unpack_h, div_9);

			unpack_l = _mm_cvtps_epi32(div_l);
			unpack_h = _mm_cvtps_epi32(div_h);

			result = _mm_packs_epi32(unpack_l, unpack_h);

			_mm_store_si128((__m128i*)dst + x + y * (width / 8), result);

		}
	}
	sse_result.convertTo(sse_result, CV_8UC1);
	return sse_result;
}



