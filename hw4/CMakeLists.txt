cmake_minimum_required(VERSION 3.16)
project( hw3 )
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( hw3 ./src/hw3.cpp )
target_link_libraries( hw3  OpenMP::OpenMP_CXX ${OpenCV_LIBS})


# set(IPP_DIR /opt/intel/oneapi/ipp/2021.5.2/lib/cmake/ipp)
# find_package(IPP REQUIRED PATHS ${IPP_DIRS})
# set(IPP_INCLUDE_DIRS /opt/intel/oneapi/ipp/2021.5.2/include)
# include_directories(${IPP_INCLUDE_DIRS})
# find_library(IPP_LIBRARIES /opt/intel/oneapi/ipp/2021.5.2/lib)
# set_target_properties(IPP::IPP PROPERTIES INTERFACE_INCLUDE_DIRECTOREIS "${IPP_INCLUDE_DIRS}" IMPORTED_LOCATION "${IPP_LIBRARIES}")


