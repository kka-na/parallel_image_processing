#ifndef HW5Mean_h
#define HW5Mean_h

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

class HW5Mean
{
public:
	Mat srcImg;
	int width;
	int height;
	const int k = 1;
	const int kscale = 3 * 3;

	HW5Mean(const Mat &_srcImg)
	{
		srcImg = _srcImg;
		width = srcImg.cols;
		height = srcImg.rows;
	}

	// intrinsics
	// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE2,SSE3,SSSE3&ig_expand=6153,4290,7350,7407,92,302,5029,5171,6929,6249,7350,6249,7407,7407,7350
	Mat myMean()
	{
		Mat dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
		// total width to multiple of 16
		const int nx0 = (width + 16) & ~15;
		// short -> 16 bit -> max value -> 32767
		const __m128i vkscale = _mm_set1_epi16(SHRT_MAX / kscale);

		for (int y = 1; y < height - 1; y++)
		{
			const uint8_t *const prev = srcImg.ptr(y - 1);
			const uint8_t *const curr = srcImg.ptr(y);
			const uint8_t *const next = srcImg.ptr(y + 1);
			uint8_t *const dst = dstImg.ptr(y);

			__m128i prev_lower, curr_upper, curr_lower, test;
			__m128i _upper, _lower;

			// Broadcast 16-bit integer '0' to all elements of dst
			prev_lower = _mm_set1_epi16(0);
			test = _mm_set1_epi16(0);
			Loading(0, prev, curr_upper, curr_lower); //-> initialize (prev+0) + 0
			Loading(0, curr, _upper, _lower);		  //-> initialize (curr+0) + 0
			// Add packed 16-bit integers in a and b, and store results in dst
			curr_upper = _mm_add_epi16(curr_upper, _upper); //->curr_upper update
			curr_lower = _mm_add_epi16(curr_lower, _lower); //->curr_lower update
			Loading(0, next, _upper, _lower);				//-> initialize (next+0) + updated _upper, _lower
			curr_upper = _mm_add_epi16(curr_upper, _upper); //->curr_upper update
			curr_lower = _mm_add_epi16(curr_lower, _lower); //->curr_lower update

			for (int x = 0; x < nx0; x += 16)
			{
				__m128i next_upper, next_lower, upper, lower;
				Loading((x + 16), prev, next_upper, next_lower); //-> initialize (prev+16) + 0
				Loading((x + 16), curr, _upper, _lower);		 //-> initialize (curr+16) + updated _upper, _lower
				next_upper = _mm_add_epi16(next_upper, _upper);	 //->next_upper update
				next_lower = _mm_add_epi16(next_lower, _lower);	 //->next_lower update
				Loading((x + 16), next, _upper, _lower);		 //-> initialize (curr+16)+ updated _upper, _lower
				next_upper = _mm_add_epi16(next_upper, _upper);	 //->next_upper update
				next_lower = _mm_add_epi16(next_lower, _lower);	 //->next_lower update

				// Concatenate 16-byte blocks in a and b into 32-byte temporary result, shift the result right by size bytes, and store the low 16 bytsd in dst
				upper = _mm_add_epi16(curr_upper, _mm_alignr_epi8(curr_upper, prev_lower, 14)); // 16-2
				lower = _mm_add_epi16(curr_lower, _mm_alignr_epi8(curr_lower, curr_upper, 14));
				upper = _mm_add_epi16(upper, _mm_alignr_epi8(curr_lower, curr_upper, 2));
				lower = _mm_add_epi16(lower, _mm_alignr_epi8(next_upper, curr_lower, 2));

				// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers.
				upper = _mm_mulhrs_epi16(upper, vkscale);
				lower = _mm_mulhrs_epi16(lower, vkscale);

				// Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturations, and store the result in dst
				__m128i result = _mm_packus_epi16(upper, lower);
				// Store 128-bits of integer data from "result" into memory " dst + x ".
				_mm_storeu_si128((__m128i *)(dst + x), result);

				prev_lower = curr_lower;
				curr_upper = next_upper;
				curr_lower = next_lower;
			}
		}
		return dstImg;
	}

	void Loading(const ssize_t offset, const uint8_t *const src, __m128i &_upper, __m128i &_lower)
	{
		// Load 128bit of integer data from memory into dst
		const __m128i v = _mm_loadu_si128((__m128i *)(src + offset));
		// Unpack and interleave 8-bit integer from the low half of a and b, and store the results in dst
		// Return vector of type __m128i with all elements set to zero
		_upper = _mm_unpacklo_epi8(v, _mm_setzero_si128()); // ->8byte
		// Unpack and interleave 8-bit integer from the high half of a and b, and store the results in dst
		_lower = _mm_unpackhi_epi8(v, _mm_setzero_si128()); //-> 8byte
	}
};

#endif
