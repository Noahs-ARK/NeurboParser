// Copyright (c) 2012-2015 Andre Martins, const string &formalism
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "Utils.h"
#include "NeuralSemanticPipe.cpp"



using namespace std;

void TrainNeuralSemanticParser();
void TestNeuralSemanticParser();

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
	google::ParseCommandLineFlags(&argc, &argv, true);
	srand(31415);
	if (FLAGS_train) {
		LOG(INFO) << "Training semantic parser..." << endl;
		TrainNeuralSemanticParser();
	} else if (FLAGS_test ) {
		LOG(INFO) << "Running semantic parser..." << endl;
		TestNeuralSemanticParser();
	}
	return 0;
}

void TrainNeuralSemanticParser() {
	int time;
	timeval start, end;
	gettimeofday(&start, NULL);

	SemanticOptions *semantic_options = new SemanticOptions;
	semantic_options->Initialize();
	NeuralSemanticPipe *pipe = new NeuralSemanticPipe(semantic_options);

	pipe->Initialize();
    if (semantic_options->prune_basic() && semantic_options->train_pruner()) {
        LOG(INFO) << "Training the pruner...";
        SemanticOptions *pruner_options = new SemanticOptions;
        *pruner_options = *semantic_options;
        pruner_options->CopyPrunerFlags();
        pruner_options->Initialize();
        NeuralSemanticPipe *pruner_pipe = NULL;
        pruner_pipe = new NeuralSemanticPipe(pruner_options);
        pruner_pipe->Initialize();
        pruner_pipe->Train();
        pipe->SetPrunerParameters(pruner_pipe->GetParameters());
        pruner_pipe->SetParameters(NULL);
        delete pruner_pipe;
        pipe->SavePruner(semantic_options->GetPrunerModelFilePath());
        return;
    }
	pipe->NeuralTrain();
	delete pipe;
	delete semantic_options;
	gettimeofday(&end, NULL);
	time = diff_ms(end, start);
	LOG(INFO) << "Training took " << static_cast<double>(time) / 1000.0
		<< " sec." << endl;
}

void TestNeuralSemanticParser() {
	int time;
	timeval start, end;
	gettimeofday(&start, NULL);
	SemanticOptions *semantic_options = new SemanticOptions;
	semantic_options->Initialize();
	NeuralSemanticPipe *pipe = new NeuralSemanticPipe(semantic_options);
	pipe -> Initialize();
	pipe->NeuralTest();
	delete pipe;
	delete semantic_options;
	gettimeofday(&end, NULL);
	time = diff_ms(end, start);

	LOG(INFO) << "Testing took " << static_cast<double>(time) / 1000.0
		<< " sec." << endl;
}
