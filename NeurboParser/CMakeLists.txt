project(neurboparser)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -Ofast -march=native -g")
enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/util
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sequence
        ${CMAKE_CURRENT_SOURCE_DIR}/src/classifier
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dependency_labeler
        ${CMAKE_CURRENT_SOURCE_DIR}/src/parser
        ${CMAKE_CURRENT_SOURCE_DIR}/../dynet
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/include/eigen3)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/lib)
set(WITH_EIGEN_BACKEND 1)
set(EXCUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(../dynet/dynet dynet)
add_subdirectory(src/util)
add_subdirectory(src/sequence)

add_subdirectory(src/classifier)
add_subdirectory(src/parser)
add_subdirectory(src/semantic_parser)

ADD_EXECUTABLE(neurboparser src/semantic_parser/NeurboParser.cpp)

target_link_libraries(neurboparser dynet pthread gflags ad3 glog
        classifier util sequence parser semantic_parser)
