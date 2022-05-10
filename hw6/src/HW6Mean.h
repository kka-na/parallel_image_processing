#ifndef HW6Mean_h
#define HW6Mean_h

#include <iostream>
// Header files for SSSE3
#include <tmmintrin.h>
#include <climits>
/*
128-bit ( 16bytes )
| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
|short|short|short|short|short|short|short|short|
*/
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class HW6Mean
{
public:
	uchar *srcData;
	uchar *dstData;
	int width;
	int height;
	const int k = 1;
	const int kscale = 3 * 3;

	HW6Mean(const Mat &_srcImg)
	{
		Mat srcImg = _srcImg;
		width = srcImg.cols;
		height = srcImg.rows;
		srcData = srcImg.data;
		dstData = (uchar *)_mm_malloc(height * width * sizeof(uchar), 8);
	}

	Mat myMean()
	{
		__m128i temp0, temp1, temp2;
		__m128i row00, row01, row02, row10, row11, row12, row20, row21, row22;
		__m128i row000, row001, row010, row011, row020, row021, row100, row101, row110, row111, row120, row121, row200, row201, row210, row211, row220, row221;
		__m128i WtoDW00, WtoDW01, WtoDW10, WtoDW11;
		__m128 ps00, ps01, ps10, ps11;
		__m128i w0, w1, w;

		__m128i cof1 = _mm_set1_epi16(0);
		__m128 nine = _mm_set_ps1(9.0f);

		int newWidth = width / 16;
		for (int y = 0; y < height - 2; y++)
		{
			__m128i *dstLast = (__m128i *)dstData + (y * newWidth) + newWidth;
			for (int x = 0; x < newWidth; x++)
			{
				row00 = _mm_load_si128((__m128i *)srcData + x + y * newWidth);
				row000 = _mm_unpacklo_epi8(row00, cof1);
				row001 = _mm_unpackhi_epi8(row00, cof1);

				temp0 = _mm_load_si128((__m128i *)srcData + x + y * newWidth + 1);

				row01 = _mm_alignr_epi8(temp0, row00, 1);
				row010 = _mm_unpacklo_epi8(row01, cof1);
				row011 = _mm_unpackhi_epi8(row01, cof1);

				row02 = _mm_alignr_epi8(temp0, row00, 2);
				row020 = _mm_unpacklo_epi8(row02, cof1);
				row021 = _mm_unpackhi_epi8(row02, cof1);

				__m128i sumro00 = _mm_add_epi16(_mm_add_epi16(row000, row010), row020);
				__m128i sumro01 = _mm_add_epi16(_mm_add_epi16(row001, row011), row021);

				row10 = _mm_load_si128((__m128i *)srcData + x + (y + 1) * newWidth);
				row100 = _mm_unpacklo_epi8(row10, cof1);
				row101 = _mm_unpackhi_epi8(row10, cof1);

				temp1 = _mm_load_si128((__m128i *)srcData + x + (y + 1) * newWidth + 1);

				row11 = _mm_alignr_epi8(temp1, row10, 1);
				row110 = _mm_unpacklo_epi8(row11, cof1);
				row111 = _mm_unpackhi_epi8(row11, cof1);

				row12 = _mm_alignr_epi8(temp1, row10, 2);
				row120 = _mm_unpacklo_epi8(row12, cof1);
				row121 = _mm_unpackhi_epi8(row12, cof1);

				__m128i sumro10 = _mm_add_epi16(_mm_add_epi16(row100, row110), row120);
				__m128i sumro11 = _mm_add_epi16(_mm_add_epi16(row101, row111), row121);

				row20 = _mm_load_si128((__m128i *)srcData + x + (y + 2) * newWidth);
				row200 = _mm_unpacklo_epi8(row20, cof1);
				row201 = _mm_unpackhi_epi8(row20, cof1);

				temp2 = _mm_load_si128((__m128i *)srcData + x + (y + 2) * newWidth + 1);

				row21 = _mm_alignr_epi8(temp2, row20, 1);
				row210 = _mm_unpacklo_epi8(row21, cof1);
				row211 = _mm_unpackhi_epi8(row21, cof1);

				row22 = _mm_alignr_epi8(temp2, row20, 2);
				row220 = _mm_unpacklo_epi8(row22, cof1);
				row221 = _mm_unpackhi_epi8(row22, cof1);

				__m128i sumro20 = _mm_add_epi16(_mm_add_epi16(row200, row210), row220);
				__m128i sumro21 = _mm_add_epi16(_mm_add_epi16(row201, row211), row221);

				__m128i sum0 = _mm_add_epi16(_mm_add_epi16(sumro00, sumro10), sumro20);
				__m128i sum1 = _mm_add_epi16(_mm_add_epi16(sumro01, sumro11), sumro21);

				WtoDW00 = _mm_unpacklo_epi16(sum0, cof1);
				WtoDW01 = _mm_unpackhi_epi16(sum0, cof1);
				WtoDW10 = _mm_unpacklo_epi16(sum1, cof1);
				WtoDW11 = _mm_unpackhi_epi16(sum1, cof1);

				ps00 = _mm_cvtepi32_ps(WtoDW00);
				ps01 = _mm_cvtepi32_ps(WtoDW01);
				ps10 = _mm_cvtepi32_ps(WtoDW10);
				ps11 = _mm_cvtepi32_ps(WtoDW11);

				ps00 = _mm_div_ps(ps00, nine);
				ps01 = _mm_div_ps(ps01, nine);
				ps10 = _mm_div_ps(ps10, nine);
				ps11 = _mm_div_ps(ps11, nine);

				WtoDW00 = _mm_cvtps_epi32(ps00);
				WtoDW01 = _mm_cvtps_epi32(ps01);
				WtoDW10 = _mm_cvtps_epi32(ps10);
				WtoDW11 = _mm_cvtps_epi32(ps11);

				w0 = _mm_packs_epi32(WtoDW00, WtoDW01);
				w1 = _mm_packs_epi32(WtoDW10, WtoDW11);

				w = _mm_packus_epi16(w0, w1);

				_mm_store_si128(dstLast + x, w);
			}
		}

		Mat dstImg(Size(height, width), CV_8UC1, dstData);
		return dstImg;
	}
};

#endif
