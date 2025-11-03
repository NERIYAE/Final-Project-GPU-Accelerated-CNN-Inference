cmake_minimum_required(VERSION 3.20)
project(cuconv_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 35;60;86)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(cuconv SHARED src/cuconv_lib.cu)
target_compile_options(cuconv PRIVATE -O3 -lineinfo)
target_link_libraries(cuconv PRIVATE CUDA::cudart)

add_executable(test_cuconv src/main.cpp)
target_link_libraries(test_cuconv PRIVATE cuconv CUDA::cudart)
