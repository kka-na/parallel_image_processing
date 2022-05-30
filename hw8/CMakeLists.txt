cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 OLD)
project( hw8 LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )

include_directories(/usr/local/cuda-11.4/samples/common/inc /usr/local/cuda-11.4/include)

add_executable(hw8 ./src/hw8.cpp ./src/kernel.cu)
target_link_libraries( hw8 PUBLIC ${OpenCV_LIBS})
