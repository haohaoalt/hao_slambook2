cmake_minimum_required(VERSION 3.10)
project(ch3demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(ch3demo task.cpp)

target_link_libraries(ch3demo ${EIGEN3_LIBRARIES})