cmake_minimum_required(VERSION 3.16)
project( ex3 )
find_package(OpenMP REQUIRED)
add_executable( ex3 ex3.cpp )
target_link_libraries( ex3  OpenMP::OpenMP_CXX)


# set(IPP_DIR /opt/intel/oneapi/ipp/2021.5.2/lib/cmake/ipp)
# find_package(IPP REQUIRED PATHS ${IPP_DIRS})
# set(IPP_INCLUDE_DIRS /opt/intel/oneapi/ipp/2021.5.2/include)
# include_directories(${IPP_INCLUDE_DIRS})
# find_library(IPP_LIBRARIES /opt/intel/oneapi/ipp/2021.5.2/lib)
# set_target_properties(IPP::IPP PROPERTIES INTERFACE_INCLUDE_DIRECTOREIS "${IPP_INCLUDE_DIRS}" IMPORTED_LOCATION "${IPP_LIBRARIES}")


