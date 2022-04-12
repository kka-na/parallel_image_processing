# INHA Univ. Parallel Image Processing Prograsmming for Graduated Students HW 2

## Usage

```
kana@Alienware:~/Documents/Class/Parallel image processing programming/hw2/build$ cmake .. && make && ./hw2
-- Configuring done
-- Generating done
-- Build files have been written to: /home/kana/Documents/Class/Parallel image processing programming/hw2/build
Consolidate compiler generated dependencies of target hw2
[100%] Built target hw2

Parallel Image Processing Programming HW2
22212231 김가나

፨ Enter Question number to To check the answer to the problem. ፨
[ Question #1 ]
Implement Gaussian filter using IPP.
[ Question #2 ]
Impelment Median filter using IPP.
[ Question #3 ]
Test on different size of images 256, 512, 1024, 2048.
[ Question #4 ]
Compare result images, processing time between OpenCV and IPP.

፨ Quit to Enter 0 ፨
▶▷▶ 1
IPP Processing time : 0.000696 sec

፨ Quit to Enter 0 ፨
▶▷▶ 2
IPP Processing time : 0.000195 sec

፨ Quit to Enter 0 ፨
▶▷▶ 3
Using IPP
[ Checking Gaussian ]
IPPProcessing time of Image [256 x 256] : 8.5e-05 sec
IPPProcessing time of Image [512 x 512] : 0.000255 sec
IPPProcessing time of Image [1024 x 1024] : 0.000974 sec
IPPProcessing time of Image [2048 x 2048] : 0.004185 sec
[ Checking Medain ]
IPPProcessing time of Image [256 x 256] : 9.1e-05 sec
IPPProcessing time of Image [512 x 512] : 0.000173 sec
IPPProcessing time of Image [1024 x 1024] : 0.000617 sec
IPPProcessing time of Image [2048 x 2048] : 0.003756 sec

፨ Quit to Enter 0 ፨
▶▷▶ 4
Using OpenCV
[ Checking Gaussian ]
OpenCVProcessing time of Image [256 x 256] : 0.000413 sec
OpenCVProcessing time of Image [512 x 512] : 0.00011 sec
OpenCVProcessing time of Image [1024 x 1024] : 0.000967 sec
OpenCVProcessing time of Image [2048 x 2048] : 0.005737 sec
[ Checking Medain ]
OpenCVProcessing time of Image [256 x 256] : 0.000419 sec
OpenCVProcessing time of Image [512 x 512] : 0.000853 sec
OpenCVProcessing time of Image [1024 x 1024] : 0.002209 sec
OpenCVProcessing time of Image [2048 x 2048] : 0.005906 sec

፨ Quit to Enter 0 ፨
▶▷▶ 0
End Homewrok 2 ...
```

<p align="center">
  <img src="./result/IPP5x5Gaussian[512 x 512].jpg"/>
</p>
<p align="center">
  <img src="./result/IPP5x5Median[512 x 512].jpg"/>
</p>
