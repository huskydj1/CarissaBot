cmake_minimum_required(VERSION 3.17)
project(CPP_Chess_Bot)

set(CMAKE_CXX_STANDARD 20)


include_directories(${PROJECT_SOURCE_DIR})
add_executable(Chess_Bot main.cpp)