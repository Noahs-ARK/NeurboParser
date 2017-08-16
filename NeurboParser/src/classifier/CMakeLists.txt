CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(classifier)

ADD_LIBRARY(classifier Alphabet.cpp Pipe.cpp
        Dictionary.cpp Options.cpp
        Reader.cpp Writer.cpp Parameters.cpp)

target_link_libraries(classifier pthread gflags ad3 glog)


