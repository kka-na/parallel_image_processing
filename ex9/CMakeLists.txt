cmake_minimum_required(VERSION 3.16)
project( ex9 LANGUAGES CXX)

find_package( OpenCV REQUIRED )
find_package( OpenCL REQUIRED )

find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda-11.4/include)

add_executable(ex9 ex9.cpp Device.cl)

target_compile_features(ex9 PRIVATE cxx_auto_type)

target_link_libraries( ex9 PRIVATE ${CUDA_OpenCL_LIBRARY} ${OpenCV_LIBS} )