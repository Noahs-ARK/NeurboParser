CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(srl)

ADD_LIBRARY(semantic_parser
        SemanticDecoder.cpp SemanticPipe.cpp
        SemanticOptions.cpp SemanticDictionary.cpp
        SemanticInstance.cpp SemanticReader.cpp
        SemanticPart.cpp SemanticPredicate.h
        SemanticInstanceNumeric.cpp SemanticWriter.cpp FactorSemanticGraph.h
        BiLSTM.h Parser.cpp Pruner.cpp)

target_link_libraries(semantic_parser dynet pthread gflags ad3 glog
        classifier util sequence parser)