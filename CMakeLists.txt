cmake_minimum_required(VERSION 3.20.0)
project(Lab_6)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} include)
add_library(mylib include/tracker.h src/tracker.cpp)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(mylib ${OpenCV_LIBS})