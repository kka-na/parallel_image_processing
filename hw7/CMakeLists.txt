cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 OLD)
project( hw7 LANGUAGES CXX CUDA)

find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
find_package(IPP REQUIRED)
set(IPP_LIBRARIES IPP::ippcore IPP::ippcc IPP::ipp_iw IPP::ippcv IPP::ippe IPP::ippi IPP::ipps IPP::ippvm )

include_directories(/usr/local/cuda-11.4/samples/common/inc /usr/local/cuda-11.4/include)

add_executable(hw7 ./src/hw7.cpp ./src/kernel.cu)
target_link_libraries( hw7 PUBLIC ${IPP_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})
