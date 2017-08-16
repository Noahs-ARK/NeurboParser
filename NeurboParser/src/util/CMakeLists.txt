CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(util)

ADD_LIBRARY(util AlgUtils.cpp SerializationUtils.cpp
        StringUtils.cpp TimeUtils.cpp)

target_link_libraries(util pthread gflags ad3 glog)


