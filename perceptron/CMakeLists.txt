cmake_minimum_required(VERSION 3.23)
project(perceptron)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")

add_executable(perceptron main.cpp)
