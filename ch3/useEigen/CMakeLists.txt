cmake_minimum_required(VERSION 3.10)
project(useEigen)


include_directories("/usr/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)