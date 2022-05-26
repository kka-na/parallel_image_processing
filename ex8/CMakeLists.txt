cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 OLD)
project( ex8 LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )

include_directories(/usr/local/cuda-11.4/samples/common/inc /usr/local/cuda-11.4/include)

add_executable(ex8 ex8.cpp kernel.cu)
target_link_libraries( ex8 ${OpenCV_LIBS})
