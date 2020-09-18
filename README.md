#### Required software

The following software must be installed prior to building this software:

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries (tested with version 1.62.0)
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended. tested with development version, changeset 9647:9464b6f3131c)
 * [CMake](http://www.cmake.org/) (tested with version 3.6.2)

#### Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

#### Building cnn

    cd cnn
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

If cmake or make have trouble finding boost or eigen, you can hardcode the appropriate directories into `CMakeFiles.txt`.
For example:

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                         ${PROJECT_SOURCE_DIR}/external/easyloggingpp/src
+                 /home/samt/lib/eigen/include/eigen3)

    set(Boost_REALPATH ON)
+    set(BOOST_ROOT /opt/tools/boost/1.62.0)

    # then run source ~/.bashrc to have those environment variable effective immediately
+     link_directories( /opt/tools/boost/1.62.0/lib)

Then comment out:
 
    # look for Eigen
    #find_package(Eigen3 REQUIRED)
    #include_directories(${EIGEN3_INCLUDE_DIR})


#### Thanks Swabha for the above part for installing cnn.

I put the cnn code into this repository. Please follow the above to install it into './cnn'. It is not recommended that you use the newest version, since they recently changed it into 'dynet' with tons of changes of usage. (By the way, this is terrible since I learnt how to use cnn only two days ago.) 

#### To fetch all the required libs:

If you are on MacOSX, edit the file `install_deps.sh` to set `RUN_AUTOTOOLS=true`.
Then run

	./install_deps.sh

It puts all the libs into './deps/' There might be problems with glog (it may or may not happen depending on your platform). If they do happen, please manually get glog and put it into the corresponding directory.
Remove `deps/local/include/Eigen`, because we've already installed a newer version.  

At this point, you should be able to use cmake to build the parser.
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2


#### Running/training the parser.

There are instructions o preparing the data under './semeval2015_data'. There is a pretrained pruner under './model'. When training the parser, you can use the following arguments:

	--train
	--train_epochs=50
	--file_model=path_to_your_model
	--file_train=path_to_Turboparser/semeval2015_data/dm/data/english/english_id_dm_augmented_train.sdp
	--srl_labeled=true
	--srl_deterministic_labels=true
	--srl_use_dependency_syntactic_features=true
	--srl_prune_labels_with_senses=true
	--srl_prune_labels=true
	--srl_prune_distances=true
	--srl_prune_basic=true
	--srl_pruner_posterior_threshold=0.0001
	--srl_pruner_max_arguments=20
	--srl_use_pretrained_pruner
	--srl_file_pruner_model=path_to_Turboparser/model/pruner.model
	--form_case_sensitive=false
	--train_algorithm=svm_mira
	--train_regularization_constant=0.01
	--srl_train_cost_false_positives=0.4
	--srl_train_cost_false_negatives=0.6
	--srl_model_type=af+as+cs+gp+cp+ccp
	--srl_allow_self_loops=false
	--srl_allow_root_predicate=true
	--srl_allow_unseen_predicates=false
	--srl_use_predicate_senses=false
	--srl_file_format=sdp
	--logtostderr
	
TODO:
	Implement regularization.
	Build the Bi-LSTM encoder.	
	Wrap up the running options.
	More explorations on the trainer.
	Switch to dynet if necessary 
