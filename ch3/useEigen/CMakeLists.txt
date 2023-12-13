cmake_minimum_required(VERSION 3.10)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -fPIC")


# 添加 eigen 库
include_directories("/usr/include/eigen3")
# 添加源文件
add_executable(eigenMatrix eigenMatrix.cpp)