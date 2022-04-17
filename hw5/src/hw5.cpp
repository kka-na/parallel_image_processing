#include "SSETest.h"
#include "SerialMean.h"
#include <chrono>

void doSerialMean();
void doTest();

int main()
{
    cout << "\nParallel Image Processing Programming HW5\n22212231 김가나\n";
    cout << "\nMean Filtering Using SSE\n";


    //Serial Mean Filter
    auto start1 = std::chrono::high_resolution_clock::now();
    doSerialMean();
    auto finish1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);

    cout << "  |- Processing Time" << endl;
        cout << "     - Serial    : " << float(duration1.count()) / 1000000 << " sec" << endl;

   destroyAllWindows();
    return 0;
}

void doSerialMean(){
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src,src,Size(512,512));
    SerialMean sm(src);
    //Mat dst= sm.myMean();
    Mat dst= sm.scalarMean();
    hconcat(src,dst,dst);

    imshow("3x3 Mean Filter Test", dst);
    waitKey(0);
}

void doTest(){
    SSETest ts;
    
    short A[8] = {1,2,3,4,5,6,7,8};
    short B[8] = {1,2,3,4,5,6,7,8};
    short C[8] = {0};
    short D[8] = {0};

    //C Program
    auto start1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<8; i++){
        C[i] =A[i]+B[i];
    }
    printf("%d, %d, %d, %d, %d, %d, %d, %d\n", C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);

    auto finish1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(finish1 - start1);


    //SIMD Program
    auto start2 = std::chrono::high_resolution_clock::now();
    ts.SIMDTest(A, B, D);
    auto finish2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(finish2 - start2);

    cout << "  |- Processing Time" << endl;
        cout << "     - C    : " << float(duration1.count()) / 1000000 << " sec" << endl;
        cout << "     - SIMD : " << float(duration2.count()) / 1000000 << " sec" << endl;

}