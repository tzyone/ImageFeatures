cmake_minimum_required(VERSION 3.0)
project(CORNER_DETECT)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/../include)

aux_source_directory(${PROJECT_SOURCE_DIR}/../impl DIR_SRCS)

message("impl dir: " ${DIR_SRCS})

add_executable(ImageFeat main.cpp ${DIR_SRCS})

target_link_libraries(ImageFeat ${OpenCV_LIBS})