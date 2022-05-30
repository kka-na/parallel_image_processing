# INHA Univ. Parallel Image Processing Programming for Graduated Students HW7

## ./hw7

```
kana@Alienware:~/Documents/Class/Parallel image processing programming/hw8/build$ cmake .. && make && ./hw8
-- Configuring done
-- Generating done
-- Build files have been written to: /home/kana/Documents/Class/Parallel image processing programming/hw8/build
Consolidate compiler generated dependencies of target hw8
[ 33%] Building CXX object CMakeFiles/hw8.dir/src/hw8.cpp.o
[ 66%] Linking CXX executable hw8
[100%] Built target hw8

Parallel Image Processing Programming HW8
22212231 김가나
Gaussian Filtering Using CUDA with Vaious Set of Memory Type
Global, Shared, Constant

Image Size = 4096x4096, Kernel Size = 25x25 , Sigma Value = 25.0f

  |- Processing Time
   - Global         : 0.256908 sec
   - Shared         : 0.188849 sec
   - Constant       : 0.114101 sec

  |- Result Summary
   - Shared is 136% faster than Global
   - Constant is 225% faster than Global
```

<p align="center">
  <p>Global</p>
  <img src="./result/global.png"/>
  <p>Shared</p>
  <img src="./result/shared.png"/>
  <p>Constant</p>
  <img src="./result/constant.png"/>
</p>
