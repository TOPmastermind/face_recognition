cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(FACE_RECOGNITION)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp face_recognition.cpp)

target_link_libraries(main LINK_PRIVATE ${OpenCV_LIBS})