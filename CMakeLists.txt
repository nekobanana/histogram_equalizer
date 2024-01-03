cmake_minimum_required(VERSION 3.27)
project(histogram_equalizer CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(histogram_equalizer main.cu)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( histogram_equalizer ${OpenCV_LIBS} )

set_target_properties(histogram_equalizer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

