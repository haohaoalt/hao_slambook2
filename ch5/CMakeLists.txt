cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#Eigen
include_directories("/usr/include/eigen3")

set(OpenCV_DIR "/home/hao007/haolib/opencv4.6/install/lib/cmake/opencv4")
find_package(OpenCV 4.6.0 REQUIRED) #找 OpenCV 4.0 安装路径

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)