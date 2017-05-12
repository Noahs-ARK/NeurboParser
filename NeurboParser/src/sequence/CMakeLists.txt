CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(sequence)


#ADD_LIBRARY(Alphabet Alphabet.cpp)
#ADD_LIBRARY(Pipe Pipe.cpp)
#ADD_LIBRARY(Dictionary Dictionary.cpp)
#ADD_LIBRARY(Reader Reader.cpp)
#ADD_LIBRARY(Writer Writer.cpp)
#ADD_LIBRARY(Options Options.cpp)

ADD_LIBRARY(sequence SequenceInstance.cpp TokenDictionary.cpp)

#set(CMAKE_C_FLAGS "-g SemanticDecoder.o NeuralSemanticPipe.o SemanticOptions.o NueralTurboSemanticParser.o SemanticDictionary.o SemanticInstance.o SemanticReader.o SemanticPart.o SemanticFeatures.o SemanticInstanceNumeric.o SemanticWriter.o DependencyInstanceNumeric.o DependencyWriter.o DependencyDictionary.o DependencyInstance.o DependencyReader.o")
target_link_libraries(sequence pthread gflags ad3 glog)


