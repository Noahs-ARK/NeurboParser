CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(neurboparser)

ADD_LIBRARY(semantic_parser
        SemanticDecoder.cpp SemanticPipe.cpp
        SemanticOptions.cpp SemanticDictionary.cpp
        SemanticInstance.cpp SemanticReader.cpp
        SemanticPart.cpp
        SemanticInstanceNumeric.cpp SemanticWriter.cpp
        FactorSemanticGraph.h SemanticPredicate.h
        BiLSTM.h Parser.h Pruner.h)

target_link_libraries(semantic_parser dynet pthread gflags ad3 glog
        classifier util sequence parser)