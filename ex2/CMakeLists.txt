cmake_minimum_required(VERSION 3.16)
project( ex2 )
find_package( OpenCV REQUIRED )
find_package(IPP REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(IPP_LIBRARIES IPP::ippcore IPP::ippcc IPP::ipp_iw IPP::ippcv IPP::ippe IPP::ippi IPP::ipps IPP::ippvm )
add_executable( ex2 ex2.cpp )
target_link_libraries( ex2  PUBLIC ${IPP_LIBRARIES} ${OpenCV_LIBS})


# set(IPP_DIR /opt/intel/oneapi/ipp/2021.5.2/lib/cmake/ipp)
# find_package(IPP REQUIRED PATHS ${IPP_DIRS})
# set(IPP_INCLUDE_DIRS /opt/intel/oneapi/ipp/2021.5.2/include)
# include_directories(${IPP_INCLUDE_DIRS})
# find_library(IPP_LIBRARIES /opt/intel/oneapi/ipp/2021.5.2/lib)
# set_target_properties(IPP::IPP PROPERTIES INTERFACE_INCLUDE_DIRECTOREIS "${IPP_INCLUDE_DIRS}" IMPORTED_LOCATION "${IPP_LIBRARIES}")


