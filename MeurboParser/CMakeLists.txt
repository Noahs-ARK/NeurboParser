project(meurboparser)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g")

enable_testing()

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/util 
					${CMAKE_CURRENT_SOURCE_DIR}/src/sequence
					${CMAKE_CURRENT_SOURCE_DIR}/src/multi_classifier
					${CMAKE_CURRENT_SOURCE_DIR}/src/multi_dependency_labeler
					${CMAKE_CURRENT_SOURCE_DIR}/src/multi_parser
					${CMAKE_CURRENT_SOURCE_DIR}/../dynet 
					${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/include/eigen3)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps/local/lib)
set(WITH_EIGEN_BACKEND 1)
set(EXCUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# look for Boost
if(DEFINED ENV{BOOST_ROOT})
      set(Boost_NO_SYSTEM_PATHS ON)
  endif()
  set(Boost_REALPATH ON)
  set(BOOST_ROOT /usr/local)
  find_package(Boost 1.61.0 COMPONENTS program_options serialization REQUIRED)
  include_directories(${Boost_INCLUDE_DIR})
  set(LIBS ${LIBS} ${Boost_LIBRARIES})

  # look for Eigen
  #find_package(Eigen3 REQUIRED)
  #include_directories(${EIGEN3_INCLUDE_DIR})

  #configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

  add_subdirectory(../dynet/dynet dynet)
  add_subdirectory(src/util)
  add_subdirectory(src/sequence)
  
  add_subdirectory(src/multi_classifier)
  add_subdirectory(src/multi_parser)
  add_subdirectory(src/multi_neural_semantic_parser)
  
  ADD_EXECUTABLE(meurboparser src/multi_neural_semantic_parser/NeuralTurboSemanticParser.cpp)

		
			
  target_link_libraries(meurboparser dynet ${Boost_LIBRARIES} pthread gflags ad3 glog
	multi_classifier util sequence multi_parser multi_semantic_parser)
