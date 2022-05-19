cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 OLD)
project( ex7 LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )

include_directories(/usr/local/cuda-11.4/samples/common/inc /usr/local/cuda-11.4/include)

add_executable(ex7 ex7.cpp kernel.cu)
target_link_libraries( ex7 ${OpenCV_LIBS})
