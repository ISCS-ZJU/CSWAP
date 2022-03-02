###############################################################################
#
#  file: src/cuZFP/CMakeLists.txt 
#
###############################################################################

project(zfp)

cmake_minimum_required(VERSION 3.0)
find_package(CUDA)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu *.c)

#set(CMAKE_C_STANDARD 99)
#source_group("Include" FILES ${CURRENT_HEADERS}) 
#source_group("Source" FILES ${CURRENT_SOURCES}) 

cuda_add_library(zfp STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#cuda_add_library(zfp ${CURRENT_HEADERS} ${CURRENT_SOURCES})


#add_executable(simple simple.c zfp.h bitstream.h cuZFP.h)
#target_link_libraries(simple m)
#set_property(TARGET simple PROPERTY CXX_STANDARD 14)
