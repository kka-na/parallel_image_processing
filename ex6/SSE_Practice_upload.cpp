#include<math.h>
#include <intrin.h>
#include <iostream>
#include <windows.h>
#include <string.h>
using namespace std;

#define _UNSIGNED_UNPACK_
//#define _UNSIGNED_UNPACK_
 
int main()
{
	
// Method for Signed unpack
#ifdef _UNSIGNED_UNPACK_
	
	__m128i dst1;
	__m128i dst2;
	__m128i m0 = _mm_set_epi16(0,0,0,0,0,0,0,0);
	__m128i source = _mm_set_epi16(1,2,3,4,5,6,7,8);
	dst1 = _mm_unpacklo_epi16(source, m0);
	dst2 = _mm_unpackhi_epi16(source, m0);
	
	for(int i=0; i<4; i++){
		cout<<dst1.m128i_i32[i]<<" ";
	}

	for(int i=0; i<8; i++){
		cout<<dst2.m128i_i32[i]<<" ";
	}
  
#endif

// Method for palignr
#ifdef _PALIGNR_
	__m128i a = _mm_set_epi16(1,2,3,4,5,6,7,8);
	__m128i b = _mm_set_epi16(9,10,11,12,13,14,15,16);
	__m128i c = _mm_alignr_epi8(a, b,4);
	for(int i=0; i<8; i++)
	{
		cout<<c.m128i_i16[i]<<" ";
	}
#endif
 
// Method for Unsigned unpack
#ifdef _SIGNED_UNPACK_	
	__m128i dst1;
	__m128i dst2;
	__m128i source = _mm_set_epi16(-1,-2,-3,-4,-5,-6,-7,-8);
 
	dst1 = _mm_unpacklo_epi16(source, source);
	dst2 = _mm_unpackhi_epi16(source, source);
	dst1 = _mm_srai_epi32(dst1, 16);
	dst2 = _mm_srai_epi32(dst2, 16);

	for(int i=0; i<4; i++)
	{
		cout<<dst1.m128i_i32[i]<<", ";
	}

	for(int i=0; i<4; i++)
	{
		cout<<dst2.m128i_i32[i]<<", ";
	}

#endif
 
#ifdef _FLOAT_ABS_
	int x = 0x7fffffff;
	__m128 cof = _mm_set1_ps(*(float*)&x);
 	__m128 src = _mm_set_ps(-1, -4, -6, -255500);
	__m128 result = _mm_and_ps(cof, src);

#endif

// Method for Interleaved Pack with Saturation
#ifdef _PACK_WITH_SAT_	
	__m128i dst1;
	__m128i source1 = _mm_set_epi16(-300,-1,-298,-2,-296,-3,-294,-4);
	__m128i source2 = _mm_set_epi16(300,1,298,2,296,3,294,4);
	dst1 = _mm_packs_epi16(source1, source2);

	for(int i=0; i<15; i++)
	{
		printf("%d ", dst1.m128i_i8[i]);
	}

#endif

#ifdef _INTERLEAVED_PACK_WITH_SATURATION_	
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi32(-1,-2,-3,-4);
	__m128i source2 = _mm_set_epi32(1,2,3,4);
	dst1 = _mm_packs_epi16(source1, source1);
	dst2 = _mm_packs_epi16(source2, source2);
	dst3 = _mm_unpacklo_epi16(dst1, dst2);

	for(int i=0; i<8; i++)
	{
		printf("%d ", dst3.m128i_i16[i]);
	}
#endif

#ifdef _INTERLEAVED_PACK_WITHOUT_SATURATION_	
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi16(-300,-1,-298,-2,-296,-3,-294,-4);
	__m128i source2 = _mm_set_epi16(300,1,298,2,296,3,294,4);
	__m128i mask = _mm_set_epi8(0,0xff,0,0xff,0,0xff,0,0xff,0,0xff,0,0xff,0,0xff,0,0xff);

	dst1 = _mm_slli_epi16(source1,8);
	dst2 = _mm_and_si128(source2, mask);
	dst3 = _mm_or_si128(dst2, dst1);


	for(int i=0; i<8; i++)
	{
		printf("%d ", dst3.m128i_i16[i]);
	}
#endif

#ifdef _NON_INTERLEAVED_UNPACK_	 
	__m128i dst1;
	__m128i dst2;
	__m128i dst3;

	__m128i source1 = _mm_set_epi32(1,2,3,4);
	__m128i source2 = _mm_set_epi32(5, 6, 7, 8);

	dst1 =_mm_unpacklo_epi32(source1,source2);
	dst2 = _mm_unpackhi_epi32(source1, source2);

  
	for(int i=0; i<4; i++)
	{
		printf("%d ", dst2.m128i_i32[i]);
	}
#endif

#ifdef _EXTWORD_

	int result = 0;
	__m128i source = _mm_set_epi16(1,2,3,4,5,6,7,8);

	result = _mm_extract_epi16(source, 3);
	printf("%d\n", result);

	source = _mm_insert_epi16(source, result, 2);

	for(int i=0; i<8; i++)
	{
		printf("%d,", source.m128i_i16[i]);
	}

#endif

#ifdef _SHUFFLE_WORD_	
	__m128i dst1;
	__m128i source1 = _mm_set_epi16(7,6,5,4,3,2,1,0);

	//dst1 = _mm_shufflehi_epi16(source1, ((3<<6) | (2<<4) | (1<<2) | (1)));
	//dst1 = _mm_shuffle_epi32(dst1, ((2<<6) | (2<<4) | (2<<2) | (2)));

	// 위 주석과 똑같은 구현임.
	dst1 = _mm_shufflehi_epi16(source1, _MM_SHUFFLE(3,2,1,1));
	dst1 = _mm_shuffle_epi32(dst1, _MM_SHUFFLE(2,2,2,2));

	for(int i=0; i<8; i++)
	{
		printf("%d,", dst1.m128i_i16[i]);
	}
	
#endif


#ifdef _SHUFFLE_SWAP_6_1_	
	//__m128i dst1;
	//__m128i source1 = _mm_set_epi16(7,6,5,4,3,2,1,0);
	//dst1 = _mm_shuffle_epi32(source1, ((3<<6) | (0<<4) | (1<<2) | (2)));
	//dst1 = _mm_shufflehi_epi16(dst1, ((3<<6) | (1<<4) | (2<<2) | (0)));
	//dst1 = _mm_shuffle_epi32(dst1, ((3<<6) | (0<<4) | (1<<2) | (2)));

	// Reverse the order of the words
	 __m128i dst1;
	__m128i source1 = _mm_set_epi16(7,6,5,4,3,2,1,0);
	dst1 = _mm_shufflelo_epi16(source1, ((0<<6) | (1<<4) | (2<<2) | 3));
	//dst1 = _mm_shufflehi_epi16(dst1, ((0<<6) | (1<<4) | (2<<2) | 3));
	//dst1 = _mm_shuffle_epi32(dst1, ((1<<6) | (0<<4) | (3<<2) | (2))); 

	// SSSE3 shffle instruction
	//__m128i dst1;
	//__m128i source = _mm_setr_epi8(10,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	//__m128i cof = _mm_setr_epi8(8,0,9,1,10,2,11,3,12,4,13,5,14,6,15,7);// 이 순서대로 데이터를 정렬
	//dst1 = _mm_shuffle_epi8(source,cof);
 
	for(int i=0; i<16; i++)
	{
		printf("%d,", dst1.m128i_i8[i]);
	}
#endif
 
	return 0;
}