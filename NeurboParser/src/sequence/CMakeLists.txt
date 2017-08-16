CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(sequence)

ADD_LIBRARY(sequence SequenceInstance.cpp TokenDictionary.cpp)

target_link_libraries(sequence pthread gflags ad3 glog)


