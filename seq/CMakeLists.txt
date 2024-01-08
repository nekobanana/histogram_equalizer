cmake_minimum_required(VERSION 3.27)
project(histogram_equalizer_seq)

set(CMAKE_CXX_STANDARD 17)

add_executable(histogram_equalizer_seq main.cpp)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(DCMAKE_BUILD_TYPE Debug)
set(DCMAKE_BUILD_TYPE Release)

target_link_libraries( histogram_equalizer_seq ${OpenCV_LIBS} )