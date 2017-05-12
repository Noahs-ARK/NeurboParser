CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(neurboparser)

ADD_LIBRARY(semantic_parser 
	SemanticDecoder.cpp NeuralSemanticPipe.cpp 
	SemanticOptions.cpp SemanticDictionary.cpp
	SemanticInstance.cpp SemanticReader.cpp 
	SemanticPart.cpp SemanticFeatures.cpp 
	SemanticInstanceNumeric.cpp SemanticWriter.cpp)

target_link_libraries(semantic_parser dynet ${Boost_LIBRARIES} pthread gflags ad3 glog 
	classifier util sequence parser)
		



