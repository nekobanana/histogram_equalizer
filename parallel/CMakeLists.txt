cmake_minimum_required(VERSION 3.27)
project(histogram_equalizer CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(histogram_equalizer main.cu
        Scanner.cuh)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(DCMAKE_BUILD_TYPE Debug)
set(DCMAKE_BUILD_TYPE Release)
#target_compile_options(histogram_equalizer PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

target_link_libraries( histogram_equalizer ${OpenCV_LIBS} )

set_target_properties(histogram_equalizer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

