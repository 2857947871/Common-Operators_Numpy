cmake_minimum_required(VERSION 3.11)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(build 3-1.Conv2D.cpp)

target_link_libraries(build PUBLIC ${OpenCV_LIBS})