cmake_minimum_required(VERSION 3.16)
project(camera_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TensorRT_INCLUDE_DIRS /usr/include /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)

add_executable(camera_tensorrt
    src/main.cpp
    src/trt_detector.cpp
    src/roi_alarm.cpp
)

target_include_directories(camera_tensorrt PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_directories(camera_tensorrt PRIVATE
    ${TensorRT_LIBRARY_DIRS}
)

target_link_libraries(camera_tensorrt PRIVATE
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
)

if (OpenCV_VERSION VERSION_GREATER_EQUAL 4.7.0)
    add_definitions(-DBATCHED_NMS)
endif ()
