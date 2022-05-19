# INHA Univ. Parallel Image Processing Programming for Graduated Students EX7

## ./ex7

```
kana@Alienware:~/Documents/Class/Parallel image processing programming/ex7/build$ cmake .. && make
-- Configuring done
-- Generating done
-- Build files have been written to: /home/kana/Documents/Class/Parallel image processing programming/ex7/build
Consolidate compiler generated dependencies of target ex7
[ 33%] Building CUDA object CMakeFiles/ex7.dir/kernel.cu.o
/home/kana/Documents/Class/Parallel image processing programming/ex7/kernel.cu(31): warning: variable "border" was declared but never referenced

/home/kana/Documents/Class/Parallel image processing programming/ex7/kernel.cu: In function ‘void gpu_Gabor(float*, float*, int, int, float*, int)’:
/home/kana/Documents/Class/Parallel image processing programming/ex7/kernel.cu:97:23: warning: ‘cudaError_t cudaThreadSynchronize()’ is deprecated [-Wdeprecated-declarations]
   97 |     cudaThreadSynchronize();
      |                       ^
/usr/local/cuda-11.4/include/cuda_runtime_api.h:1048:46: note: declared here
 1048 | extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);
      |                                              ^~~~~~~~~~~~~~~~~~~~~
/home/kana/Documents/Class/Parallel image processing programming/ex7/kernel.cu:97:23: warning: ‘cudaError_t cudaThreadSynchronize()’ is deprecated [-Wdeprecated-declarations]
   97 |     cudaThreadSynchronize();
      |                       ^
/usr/local/cuda-11.4/include/cuda_runtime_api.h:1048:46: note: declared here
 1048 | extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);
      |                                              ^~~~~~~~~~~~~~~~~~~~~
[ 66%] Linking CXX executable ex7
[100%] Built target ex7
kana@Alienware:~/Documents/Class/Parallel image processing programming/ex7/build$ ./ex7
4131    4048    4048
-0.000299       -0.006005       -0.016323       -0.006005       -0.000299
-0.003522       -0.070738       -0.192286       -0.070738       -0.003522
-0.000000       -0.000000       -0.000000       0.000000        0.000000
0.003522        0.070738        0.192286        0.070738        0.003522
0.000299        0.006005        0.016323        0.006005        0.000299
Processing Time(8bit): 134873.000000 msec
```
