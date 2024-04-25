cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(vector_addition LANGUAGES CXX CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures to compile for")
endif()

file(GLOB CUDA_VECADD_SOURCES "src/*.cu")
add_executable(vector_addition ${CUDA_VECADD_SOURCES})
# Set include directories
target_include_directories(vector_addition PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
