cmake_minimum_required(VERSION 3.15)
project(CarissaBot)

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR})
add_executable(Demo demo.cpp)
add_executable(Chess_Bot main.cpp)

#[[
For MSVC:
Environmental Variables:
PATH: D:\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86
]]