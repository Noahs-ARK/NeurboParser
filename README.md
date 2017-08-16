NeurboParser
=================

This software implements the semantic dependency parsers described in [1]: './NeurboParser' corresponds to the basic model, and './MeurboParser' to the multitask models.

## Required software

The following software are needed to build the parser:

 * A C++ compiler supporting the [C++11 language standard] (https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries (tested with version 1.61.0)
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended. tested with development version, changeset 9647:9464b6f3131c)
 * [CMake](http://www.cmake.org/) (tested with version 3.6.2)
 * [DyNet](https://github.com/clab/dynet)

## Checking out the project for the first time

The first time you clone the repository, you need to sync the `dynet/` submodule.

    git clone https://github.com/clab/dynet.git

## To fetch all the required libs:
	
	./install_deps.sh

## To build the parser
	
	mkdir -p NeurboParer/build
	cd NeurboParser/build
	cmake ..; make -j4
	cd ../..

## Running/training the parser

	Follow the instructions under './semeval2015_data' to prepare the data. You can use the scripts in './NeurboParser' to train/evaluate the parser.
	
## To replicate the results

	You will first need to put 100-dimensional pretrained GloVe embedding [3] under 'embedding/'. Default hyperparameter values in the scripts are used in [1].
	
	The current version of DyNet uses some difference strategies to deal with numerical issues from older versions. Based on our expericence, we expect the current parser has slightly better evaluation numbers on benchmark datasets than those describsed in [1].
	
	We are still working on adopting new version of DyNet for the Multitask models.  
	
## References
	
	[1] Hao Peng, Sam Thomson, and Noah A. Smith. 2017. 
	Deep Multitask Learning for Semantic Dependency Parsing
	In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).
	
	[2] André F. T. Martins, Miguel B. Almeida, Noah A. Smith. 2013. 
	Turning on the Turbo: Fast Third-Order Non-Projective Turbo Parsers. 
	In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).
	
	[3] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. 
	GloVe: Global Vectors for Word Representation. 
	In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP).
