# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 3.10)
# 声明一个 cmake 工程
project(HelloSLAM)
# 设置编译模式
set(CMAKE_BUILD_TYPE "Debug")


# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(helloSLAM helloSLAM.cpp)

# 添加hello库
# 语法：add_library( 库名 源代码文件 ）
add_library(hello libHelloSLAM.cpp)
# 共享库
# 语法：add_library( 库名 SHARED 源代码文件 ）
add_library(hello_shared SHARED libHelloSLAM.cpp)


add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello_shared)