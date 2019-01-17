cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(surf_cuda LANGUAGES CXX CUDA)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CUDA_FLAGS="-arch=sm_61")

include_directories(${PROJECT_SOURCE_DIR}/include)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
add_subdirectory(src)  
add_subdirectory(test)
