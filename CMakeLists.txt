cmake_minimum_required(VERSION 3.10)
project(FurierTrans)

add_executable(FourierTransform FourierTransformation.c)

set(kernels "FourierKernel.cl")
foreach(kernel ${kernels})
  configure_file(${kernel} ${kernel} COPYONLY)
  message(${kernel})
endforeach()

find_package(OpenCL REQUIRED)

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 

target_link_libraries(FourierTransform ${OpenCL_LIBRARY})