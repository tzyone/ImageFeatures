cmake_minimum_required(VERSION 3.0)
project(CORNER_DETECT)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR})

add_executable(ImageFeat main.cpp canny.cpp harris.cpp
                         harrisLaplace.cpp hogBlob.cpp
                         sobel.cpp)

target_link_libraries(ImageFeat ${OpenCV_LIBS})