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

#include <stdlib.h>
#include <iostream>
#include <glog/logging.h>
#include "Utils.h"
#include "SemanticPipe.cpp"

using namespace std;

void TrainNeurboParser();

void TestNeurboParser();

int main(int argc, char **argv) {
    dynet::initialize(argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    srand(31415);
    if (FLAGS_train) {
        LOG(INFO) << "Training semantic parser..." << endl;
        TrainNeurboParser();
    } else if (FLAGS_test) {
        LOG(INFO) << "Running semantic parser..." << endl;
        TestNeurboParser();
    }
    return 0;
}

void TrainNeurboParser() {
    int time;
    timeval start, end;
    gettimeofday(&start, NULL);

    SemanticOptions *semantic_options = new SemanticOptions;
    semantic_options->Initialize();
    SemanticPipe *pipe = new SemanticPipe(semantic_options);

    pipe->Initialize();
    pipe->SaveModelFile();
    if (semantic_options->prune_basic() && semantic_options->train_pruner()) {
        LOG(INFO) << "Training the pruner...";
        pipe->TrainPruner();
        semantic_options->train_pruner_off();
    }
    pipe->Train();
    delete pipe;
    delete semantic_options;
    gettimeofday(&end, NULL);
    time = diff_ms(end, start);
    LOG(INFO) << "Training took " << static_cast<double>(time) / 1000.0
              << " sec." << endl;
}

void TestNeurboParser() {
    int time;
    timeval start, end;
    gettimeofday(&start, NULL);
    SemanticOptions *semantic_options = new SemanticOptions;
    semantic_options->Initialize();
    SemanticPipe *pipe = new SemanticPipe(semantic_options);
    pipe->Pipe::Initialize();
    pipe->LoadModelFile();
    semantic_options->train_pruner_off();
    pipe->Test();
    delete pipe;
    delete semantic_options;
    gettimeofday(&end, NULL);
    time = diff_ms(end, start);

    LOG(INFO) << "Testing took " << static_cast<double>(time) / 1000.0
              << " sec." << endl;
}
