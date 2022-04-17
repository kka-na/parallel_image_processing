#ifndef SSETest_H
#define SSETest_H

#include <iostream>
#include <emmintrin.h>

using namespace std;

class SSETest
{
public:
    void SIMDTest(short *A, short *B, short *C){
        __m128i xmmA = _mm_loadu_si128((__m128i*)A);
        __m128i xmmB = _mm_loadu_si128((__m128i*)B);
        __m128i xmmC= _mm_add_epi16(xmmA,xmmB);
        _mm_storeu_si128((__m128i*)C,xmmC);
        printf("%d, %d, %d, %d, %d, %d, %d, %d\n", C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);
    }
};
#endif
