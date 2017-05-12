// Copyright (c) 2012-2015 Andre Martins
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

#ifndef NeuralSemanticPipe_H_
#define NeuralSemanticPipe_H_

#include <hash_map>
#include "Pipe.h"
#include "SemanticOptions.h"
#include "SemanticReader.h"
#include "SemanticDictionary.h"
#include "TokenDictionary.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticWriter.h"
#include "SemanticPart.h"
#include "SemanticFeatures.h"
#include "SemanticDecoder.h"
#include "BiLSTM.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

using namespace dynet::expr;
using namespace std;


const int CROSSFORM_CUTOFF = 0;

class NeuralSemanticPipe : public Pipe {
public:
    NeuralSemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
        task1_token_dictionary_ = NULL;
        task2_token_dictionary_ = NULL;
        task3_token_dictionary_ = NULL;
        task1_dependency_dictionary_ = NULL;
        task2_dependency_dictionary_ = NULL;
        task3_dependency_dictionary_ = NULL;
        task1_pruner_parameters_ = NULL;
        task2_pruner_parameters_ = NULL;
        task3_pruner_parameters_ = NULL;
        pruner_parameters_ = NULL;
        train_pruner_ = false;
        trainer = NULL;
        model = NULL;
        parser = NULL;
    }

    virtual ~NeuralSemanticPipe() {
        delete task1_token_dictionary_;
        delete task2_token_dictionary_;
        delete task3_token_dictionary_;
        delete task1_dependency_dictionary_;
        delete task2_dependency_dictionary_;
        delete task3_dependency_dictionary_;
        delete task1_pruner_parameters_;
        delete task2_pruner_parameters_;
        delete task3_pruner_parameters_;
        delete trainer;
        delete parser;
        delete model;
    }

    SemanticReader *GetSemanticReader(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        if (formalism == "task1") {
            return static_cast<SemanticReader *> (task1_reader_);
        } else if (formalism == "task2") {
            return static_cast<SemanticReader *> (task2_reader_);
        } else if (formalism == "task3") {
            return static_cast<SemanticReader *> (task3_reader_);
        }
    }

    SemanticReader *SetSemanticReader(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        if (formalism == "task1") {
            reader_ = task1_reader_;
        } else if (formalism == "task2") {
            reader_ = task2_reader_;
        } else if (formalism == "task3") {
            reader_ = task3_reader_;
        }
    }

    SemanticDictionary *GetSemanticDictionary(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        if (formalism == "task1") {
            return static_cast<SemanticDictionary *>(task1_dictionary_);
        } else if (formalism == "task2") {
            return static_cast<SemanticDictionary *>(task2_dictionary_);
        } else if (formalism == "task3") {
            return static_cast<SemanticDictionary *>(task3_dictionary_);
        }
    }

    SemanticReader *GetSemanticReader() {
        return static_cast<SemanticReader *>(reader_);
    }

    SemanticDictionary *GetSemanticDictionary() {
        return static_cast<SemanticDictionary *>(dictionary_);
    }

    void SetSemanticDictionary(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        if (formalism == "task1") {
            dictionary_ = task1_dictionary_;
        } else if (formalism == "task2") {
            dictionary_ = task2_dictionary_;
        } else if (formalism == "task3") {
            dictionary_ = task3_dictionary_;
        }
    }

    SemanticDecoder *GetSemanticDecoder() {
        return static_cast<SemanticDecoder *>(decoder_);
    }

    SemanticOptions *GetSemanticOptions() {
        return static_cast<SemanticOptions *>(options_);
    }

    void Initialize() {
        Pipe::Initialize();
        task1_pruner_parameters_ = new Parameters;
        task2_pruner_parameters_ = new Parameters;
        task3_pruner_parameters_ = new Parameters;
    }

    void NeuralInitialize() {
        model = new dynet::Model();
        SemanticOptions *semantic_options = GetSemanticOptions();
        if (semantic_options->trainer() == "adadelta")
            trainer = new dynet::AdadeltaTrainer(*model);
        else if (semantic_options -> trainer() == "adam") {
            trainer = new dynet::AdamTrainer(*model, 0.001, 0.9, 0.9, 1e-8);
        } else if (semantic_options->trainer() == "sgd_momentum") {
            trainer = new dynet::MomentumSGDTrainer(*model);
            trainer->eta_decay = 0.02;
        } else if (semantic_options->trainer() == "sgd") {
            trainer = new dynet::SimpleSGDTrainer(*model);
            trainer->eta_decay = 0.02;
        } else {
            CHECK(false) << "Unsupported trainer. Giving up...";
        }
		trainer->clip_threshold = 1.0;
        int task1_num_roles = GetSemanticDictionary("task1") -> GetNumRoles();
        int task2_num_roles = GetSemanticDictionary("task2") -> GetNumRoles();
        int task3_num_roles = GetSemanticDictionary("task3") -> GetNumRoles();

        if (semantic_options -> output_term() == "shared1"){
            parser = new Shared1<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles, model);
        } else if (semantic_options -> output_term() == "shared3"){
            parser = new Shared3<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles, model);
        } else if (semantic_options -> output_term() == "freda1") {
            parser = new Freda1<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles, model);
        } else if (semantic_options -> output_term() == "freda3") {
            parser = new Freda3<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles, model);
        }
        else {
            CHECK(false)<< "Unsupported output term. Giving up."<<endl;
        }
        parser->InitParams(model);
    }

    void SetPruner(const string &formalism) {
        if (formalism == "task1") {
            pruner_parameters_ = task1_pruner_parameters_;
        }
        else if (formalism == "task2") {
            pruner_parameters_ = task2_pruner_parameters_;
        }
        else if (formalism == "task3") {
            pruner_parameters_ = task3_pruner_parameters_;
        }
        else {
            LOG(INFO) <<"Unsupported formalism: " <<formalism <<". Giving up..."<<endl;
            CHECK(false);
        }
    }

    void SetInstances(const string &formalism) {
        if (formalism == "task1") {
            instances_ = task1_instances_;
        }
        else if (formalism == "task2") {
            instances_ = task2_instances_;
        }
        else if (formalism == "task3") {
            instances_ = task3_instances_;
        }
        else {
            LOG(INFO) <<"Unsupported formalism: " <<formalism <<". Giving up..."<<endl;
            CHECK(false);
        }
    }

    void SetPrunerParameters(Parameters *pruner_parameters, const string &formalism) {
        if (formalism == "task1") {
            task1_pruner_parameters_ = pruner_parameters;
        }
        else if (formalism == "task2") {
            task2_pruner_parameters_ = pruner_parameters;
        }
        else if (formalism == "task3") {
            task3_pruner_parameters_ = pruner_parameters;
        }
        else {
            LOG(INFO) <<"Unsupported formalism: " <<formalism <<". Giving up..."<<endl;
            CHECK(false);
        }
    }

    void LoadPretrainedEmbedding();

    void PruneCrossForm(const vector<int> &idxs);

    void CrossFormLabelOn (const int &idx) {
        if (allowed_crossform_labeles_.find(idx) != allowed_crossform_labeles_.end()) {
            allowed_crossform_labeles_[idx] += 1;
        }
        else {
            allowed_crossform_labeles_[idx] = 1;
        }
    }

    bool IsCrossFormLabelAllowed (const int &idx) {
        return (allowed_crossform_labeles_.find(idx) != allowed_crossform_labeles_.end()
        && allowed_crossform_labeles_[idx] >= CROSSFORM_CUTOFF);
    }

    void Train(const string &formalism);

    void NeuralTrain();

    double NeuralTrainEpoch(const vector<int> & idxs, int epoch);

    void NeuralTest();

    void NeuralRun(double &unlabeled_F1, double &labeled_F1);

    void LoadNueralModel();

    void SaveNueralModel();

    void LoadPruner(const std::string &file_name, const string &formalism) {
        if (formalism == "task1") {
            FILE *fs = fopen(file_name.c_str(), "rb");
            CHECK(fs) << "Could not open model file for reading: " << file_name;
            task1_pruner_parameters_->Load(fs);
            fclose(fs);
        } else if (formalism == "task2") {
            FILE *fs = fopen(file_name.c_str(), "rb");
            CHECK(fs) << "Could not open model file for reading: " << file_name;
            task2_pruner_parameters_->Load(fs);
            fclose(fs);
        } else if (formalism == "task3") {
            FILE *fs = fopen(file_name.c_str(), "rb");
            CHECK(fs) << "Could not open model file for reading: " << file_name;
            task3_pruner_parameters_->Load(fs);
            fclose(fs);
        } else {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(false);
        }
    }

    void SavePruner(const std::string &file_name, const string &formalism) {
        if (formalism == "task1") {
            FILE *fs = fopen(file_name.c_str(), "wb");
            CHECK(fs) << "Could not open model file for writing: " << file_name;
            task1_pruner_parameters_->Save(fs);
            fclose(fs);
        } else if (formalism == "task2") {
            FILE *fs = fopen(file_name.c_str(), "wb");
            CHECK(fs) << "Could not open model file for writing: " << file_name;
            task2_pruner_parameters_->Save(fs);
            fclose(fs);
        } else if (formalism == "task3") {
            FILE *fs = fopen(file_name.c_str(), "wb");
            CHECK(fs) << "Could not open model file for writing: " << file_name;
            task3_pruner_parameters_->Save(fs);
            fclose(fs);
        }
        else {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(false);
        }
    }

protected:
    void CreateDictionary() {
        dictionary_ = new SemanticDictionary(this);
//        GetSemanticDictionary()->SetTokenDictionary(token_dictionary_);
//        GetSemanticDictionary()->SetDependencyDictionary(dependency_dictionary_);

        task1_dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary("task1")->SetTokenDictionary(task1_token_dictionary_);
        GetSemanticDictionary("task1")->SetDependencyDictionary(task1_dependency_dictionary_);

        task2_dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary("task2")->SetTokenDictionary(task2_token_dictionary_);
        GetSemanticDictionary("task2")->SetDependencyDictionary(task2_dependency_dictionary_);

        task3_dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary("task3")->SetTokenDictionary(task3_token_dictionary_);
        GetSemanticDictionary("task3")->SetDependencyDictionary(task3_dependency_dictionary_);

    }

    void CreateReader() {
        reader_ = new SemanticReader(options_);
        task1_reader_ = new SemanticReader(options_);
        task2_reader_ = new SemanticReader(options_);
        task3_reader_ = new SemanticReader(options_);
    }

    void CreateWriter() {
        task1_writer_ = new SemanticWriter(options_);
        task2_writer_ = new SemanticWriter(options_);
        task3_writer_ = new SemanticWriter(options_);
    }

    void CreateDecoder() { decoder_ = new SemanticDecoder(this); }

    Parts *CreateParts() { return new SemanticParts; }

    Features *CreateFeatures() { return new SemanticFeatures(this); }

    void CreateTokenDictionary() {
        task1_token_dictionary_ = new TokenDictionary(this);
        task2_token_dictionary_ = new TokenDictionary(this);
        task3_token_dictionary_ = new TokenDictionary(this);
    }

    void CreateDependencyDictionary() {
        task1_dependency_dictionary_ = new DependencyDictionary(this);
        task2_dependency_dictionary_ = new DependencyDictionary(this);
        task3_dependency_dictionary_ = new DependencyDictionary(this);
    }

    Parameters *GetTrainingParameters() {
        if (train_pruner_) return pruner_parameters_;
        return parameters_;
    }

    void CreateInstances(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }

        if (formalism == "task1") {
            timeval start, end;
            gettimeofday(&start, NULL);
            SemanticOptions *semantic_options = GetSemanticOptions();
            LOG(INFO) << "Creating DM instances...";
            task1_reader_->Open(semantic_options->GetTrainingFilePath(formalism));
            DeleteInstances(formalism);
            Instance *instance = task1_reader_->GetNext();
            while (instance) {
                AddInstance(formalism, instance);
                instance = task1_reader_->GetNext();
            }
            task1_reader_->Close();
            LOG(INFO) << "Number of instances: " << task1_instances_.size();
            gettimeofday(&end, NULL);
            LOG(INFO) << "Time: " << diff_ms(end, start);
        } else if (formalism == "task2") {
            timeval start, end;
            gettimeofday(&start, NULL);
            SemanticOptions *semantic_options = GetSemanticOptions();
            LOG(INFO) << "Creating PAS instances...";
            task2_reader_->Open(semantic_options->GetTrainingFilePath(formalism));
            DeleteInstances(formalism);
            Instance *instance = task2_reader_->GetNext();
            while (instance) {
                AddInstance(formalism, instance);
                instance = task2_reader_->GetNext();
            }
            task2_reader_->Close();
            LOG(INFO) << "Number of instances: " << task2_instances_.size();
            gettimeofday(&end, NULL);
            LOG(INFO) << "Time: " << diff_ms(end, start);
        } else if (formalism == "task3") {
            timeval start, end;
            gettimeofday(&start, NULL);
            SemanticOptions *semantic_options = GetSemanticOptions();
            LOG(INFO) << "Creating PSD instances...";
            task3_reader_->Open(semantic_options->GetTrainingFilePath(formalism));
            DeleteInstances(formalism);
            Instance *instance = task3_reader_->GetNext();
            while (instance) {
                AddInstance(formalism, instance);
                instance = task3_reader_->GetNext();
            }
            task3_reader_->Close();
            LOG(INFO) << "Number of instances: " << task3_instances_.size();
            gettimeofday(&end, NULL);
            LOG(INFO) << "Time: " << diff_ms(end, start);
        }
    }

    void AddInstance(const string &formalism, Instance *instance) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        Instance *formatted_instance = GetFormattedInstance(formalism, instance);
        if (formalism == "task1") {
            task1_instances_.push_back(formatted_instance);
        } else if (formalism == "task2") {
            task2_instances_.push_back(formatted_instance);
        } else if (formalism == "task3") {
            task3_instances_.push_back(formatted_instance);
        }
        if (instance != formatted_instance) delete instance;
    }

    void DeleteInstances(const string &formalism) {
        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }

        if (formalism == "task1") {
            for (int i = 0; i < task1_instances_.size(); ++i) {
                delete task1_instances_[i];
            }
            task1_instances_.clear();
            for (int i = 0; i < task1_instances_.size(); ++i) {
                delete task1_instances_[i];
            }
            task1_instances_.clear();
        } else if (formalism == "task2") {
            for (int i = 0; i < task2_instances_.size(); ++i) {
                delete task2_instances_[i];
            }
            task2_instances_.clear();
            for (int i = 0; i < task2_instances_.size(); ++i) {
                delete task2_instances_[i];
            }
            task2_instances_.clear();
        } else if (formalism == "task3") {
            for (int i = 0; i < task3_instances_.size(); ++i) {
                delete task3_instances_[i];
            }
            task3_instances_.clear();
            for (int i = 0; i < task3_instances_.size(); ++i) {
                delete task3_instances_[i];
            }
            task3_instances_.clear();
        }
    }
    void PreprocessData();
    void PreprocessData(const string &formalism);

    Instance *GetFormattedInstance(const string &formalism, Instance *instance) {
        SemanticInstanceNumeric *instance_numeric =
                new SemanticInstanceNumeric;
        instance_numeric->Initialize(*GetSemanticDictionary(formalism), static_cast<SemanticInstance *>(instance));
        return instance_numeric;
    }

    void SaveModel(FILE *fs);

    void LoadModel(FILE *fs);

    void MakeParts(Instance *instance, Parts *parts,
                   vector<double> *gold_outputs);

    void MakePartsBasic(Instance *instance, Parts *parts,
                        vector<double> *gold_outputs);

    void MakePartsBasic(Instance *instance, bool add_labeled_parts, Parts *parts,
                        vector<double> *gold_outputs);

    void MakePartsGlobal(Instance *instance, Parts *parts,
                         vector<double> *gold_outputs);

    void MakePartsCrossFormHighOrder(Instance *task1_instance, Parts *task1_parts, vector<double> *task1_gold_outputs,
                            Instance *task2_instance, Parts *task2_parts, vector<double> *task2_gold_outputs,
                            Instance *task3_instance, Parts *task3_parts, vector<double> *task3_gold_outputs,
                            Parts *parts, vector<double> *gold_outputs);

    void MakePartsArbitrarySiblings(Instance *instance,
                                    Parts *parts,
                                    vector<double> *gold_outputs);


    void MakePartsLabeledArbitrarySiblings(Instance *instance, Parts *parts, vector<double> *gold_outputs);

    void MakePartsConsecutiveSiblings(Instance *instance, Parts *parts, vector<double> *gold_outputs);

    void MakePartsGrandparents(Instance *instance, Parts *parts, vector<double> *gold_outputs);

    void MakePartsCoparents(Instance *instance, Parts *parts, vector<double> *gold_outputs);

    void MakePartsConsecutiveCoparents(Instance *instance, Parts *parts, vector<double> *gold_outputs);


    void MakeFeatures(Instance *instance, Parts *parts, bool pruner, Features *features) {
        vector<bool> selected_parts(parts->size(), true);
        MakeSelectedFeatures(instance, parts, pruner, selected_parts, features);
    }

    void MakeSelectedFeatures(Instance *instance, Parts *parts,
                              const vector<bool> &selected_parts, Features *features) {
        // Set pruner = false unless we're training the pruner.
        MakeSelectedFeatures(instance, parts, train_pruner_, selected_parts, features);
    }

    void MakeSelectedFeatures(Instance *instance,
                              Parts *parts,
                              bool pruner,
                              const vector<bool> &selected_parts,
                              Features *features);


    void ComputeScores(Instance *instance, Parts *parts, Features *features,
                       vector<double> *scores) {
        // Set pruner = false unless we're training the pruner.
        ComputeScores(instance, parts, features, train_pruner_, scores);
    }

    void ComputeScores(Instance *instance, Parts *parts, Features *features,
                       bool pruner, vector<double> *scores);

    void RemoveUnsupportedFeatures(Instance *instance, Parts *parts,
                                   bool pruner,
                                   const vector<bool> &selected_parts,
                                   Features *features);

    void RemoveUnsupportedFeatures(Instance *instance, Parts *parts,
                                   const vector<bool> &selected_parts,
                                   Features *features) {
        // Set pruner = false unless we're training the pruner.
        RemoveUnsupportedFeatures(instance, parts, train_pruner_, selected_parts,
                                  features);
    }

    void MakeFeatureDifference(Parts *parts,
                               Features *features,
                               const vector<double> &gold_output,
                               const vector<double> &predicted_output,
                               FeatureVector *difference);

    void MakeGradientStep(Parts *parts,
                          Features *features,
                          double eta,
                          int iteration,
                          const vector<double> &gold_output,
                          const vector<double> &predicted_output);

    void TouchParameters(Parts *parts, Features *features,
                         const vector<bool> &selected_parts);

    void LabelInstance(Parts *parts, const vector<double> &output,
                       Instance *instance);

    void Prune(Instance *instance, Parts *parts, vector<double> *gold_outputs,
               bool preserve_gold);

    virtual void BeginEvaluation() {
        all_num_predicted_unlabeled_arcs_ = 0;
        all_num_gold_unlabeled_arcs_ = 0;
        all_num_matched_unlabeled_arcs_ = 0;
        all_num_gold_labeled_arcs_ = 0;
        all_num_matched_labeled_arcs_ = 0;
        all_num_predicted_labeled_arcs_ = 0;
        all_num_pruned_gold_unlabeled_arcs_ = 0;
        all_num_pruned_gold_labeled_arcs_ = 0;

        task1_num_predicted_unlabeled_arcs_ = 0;
        task1_num_gold_unlabeled_arcs_ = 0;
        task1_num_matched_unlabeled_arcs_ = 0;
        task1_num_gold_labeled_arcs_ = 0;
        task1_num_matched_labeled_arcs_ = 0;
        task1_num_predicted_labeled_arcs_ = 0;
        task1_num_pruned_gold_unlabeled_arcs_ = 0;
        task1_num_pruned_gold_labeled_arcs_ = 0;

        task2_num_predicted_unlabeled_arcs_ = 0;
        task2_num_gold_unlabeled_arcs_ = 0;
        task2_num_matched_unlabeled_arcs_ = 0;
        task2_num_gold_labeled_arcs_ = 0;
        task2_num_matched_labeled_arcs_ = 0;
        task2_num_predicted_labeled_arcs_ = 0;
        task2_num_pruned_gold_unlabeled_arcs_ = 0;
        task2_num_pruned_gold_labeled_arcs_ = 0;

        task3_num_predicted_unlabeled_arcs_ = 0;
        task3_num_gold_unlabeled_arcs_ = 0;
        task3_num_matched_unlabeled_arcs_ = 0;
        task3_num_gold_labeled_arcs_ = 0;
        task3_num_matched_labeled_arcs_ = 0;
        task3_num_predicted_labeled_arcs_ = 0;
        task3_num_pruned_gold_unlabeled_arcs_ = 0;
        task3_num_pruned_gold_labeled_arcs_ = 0;

        num_tokens_ = 0;
        num_unlabeled_arcs_after_pruning_ = 0;
        num_labeled_arcs_after_pruning_ = 0;
        gettimeofday(&start_clock_, NULL);
    }

    virtual void EvaluateInstance(const string& formalism, Instance *instance, Instance *output_instance,
                                  Parts *parts, const vector<double> &gold_outputs,
                                  const vector<double> &predicted_outputs)
    {

        if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
            LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
            CHECK(1 == 0);
        }
        int num_possible_unlabeled_arcs = 0;
        int num_possible_labeled_arcs = 0;

        int num_gold_unlabeled_arcs = 0;
        int num_matched_unlabeled_arcs = 0;
        int num_predicted_unlabeled_arcs = 0;

        int num_gold_labeled_arcs = 0;
        int num_matched_labeled_arcs = 0;
        int num_predicted_labeled_arcs = 0;
        SemanticInstance *semantic_instance =
                static_cast<SemanticInstance *>(instance);
        SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
        for (int p = 0; p < semantic_instance->size(); ++p) {
            const vector<int> &senses = semantic_parts->GetSenses(p);
            for (int a = 1; a < semantic_instance->size(); ++a) {
                for (int k = 0; k < senses.size(); ++k) {
                    int s = senses[k];
                    int r = semantic_parts->FindArc(p, a, s);
                    if (r < 0) continue;
                    ++num_possible_unlabeled_arcs;
                    if (gold_outputs[r] >= 0.5) {
                        CHECK_EQ(gold_outputs[r], 1.0);
                        if (NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6)) {
                            ++num_matched_unlabeled_arcs;
                        }
                        ++num_gold_unlabeled_arcs;
                    }
                    if (predicted_outputs[r] >= 0.5) {
                        //CHECK_EQ(predicted_outputs[r], 1.0);
                        ++num_predicted_unlabeled_arcs;

                        //LOG(INFO) << semantic_instance->GetForm(a)
                        //          << " <-- "
                        //          << semantic_instance->GetForm(p);
                    }
                    if (GetSemanticOptions()->labeled()) {
                        const vector<int> &labeled_arcs =
                                semantic_parts->FindLabeledArcs(p, a, s);
                        for (int k = 0; k < labeled_arcs.size(); ++k) {
                            int r = labeled_arcs[k];
                            if (r < 0) continue;
                            ++num_possible_labeled_arcs;
                            if (gold_outputs[r] >= 0.5) {
                                CHECK_EQ(gold_outputs[r], 1.0);
                                if (NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6)) {
                                    ++num_matched_labeled_arcs;

                                    //LOG(INFO) << semantic_instance->GetForm(a)
                                    //          << " <-*- "
                                    //          << semantic_instance->GetForm(p);
                                }
                                ++num_gold_labeled_arcs;
                            }
                            if (predicted_outputs[r] >= 0.5) {
//                                CHECK_EQ(predicted_outputs[r], 1.0);
                                ++num_predicted_labeled_arcs;
                            }
                        }
                    }
                }
            }

            ++num_tokens_;
            num_unlabeled_arcs_after_pruning_ += num_possible_unlabeled_arcs;
            num_labeled_arcs_after_pruning_ += num_possible_labeled_arcs;
        }

        int num_actual_gold_arcs = 0;
        for (int k = 0; k < semantic_instance->GetNumPredicates(); ++k) {
            num_actual_gold_arcs +=
                    semantic_instance->GetNumArgumentsPredicate(k);
        }
        int missed_unlabeled = num_actual_gold_arcs - num_gold_unlabeled_arcs;
        int missed_labeled = num_actual_gold_arcs - num_gold_labeled_arcs;
        int missed = missed_unlabeled + missed_labeled;
        if (formalism == "task1") {
            task1_num_gold_unlabeled_arcs_ += num_actual_gold_arcs;
            task1_num_predicted_unlabeled_arcs_ += num_predicted_unlabeled_arcs;
            task1_num_matched_unlabeled_arcs_ += num_matched_unlabeled_arcs;

            task1_num_gold_labeled_arcs_ += num_actual_gold_arcs;
            task1_num_predicted_labeled_arcs_ += num_predicted_labeled_arcs;
            task1_num_matched_labeled_arcs_ += num_matched_labeled_arcs;

            task1_num_pruned_gold_unlabeled_arcs_ += missed_unlabeled;
            task1_num_pruned_gold_labeled_arcs_ += missed_labeled;
        } else if (formalism == "task2") {
            task2_num_gold_unlabeled_arcs_ += num_actual_gold_arcs;
            task2_num_predicted_unlabeled_arcs_ += num_predicted_unlabeled_arcs;
            task2_num_matched_unlabeled_arcs_ += num_matched_unlabeled_arcs;

            task2_num_gold_labeled_arcs_ += num_actual_gold_arcs;
            task2_num_predicted_labeled_arcs_ += num_predicted_labeled_arcs;
            task2_num_matched_labeled_arcs_ += num_matched_labeled_arcs;

            task2_num_pruned_gold_unlabeled_arcs_ += missed_unlabeled;
            task2_num_pruned_gold_labeled_arcs_ += missed_labeled;
        } else if (formalism == "task3") {
            task3_num_gold_unlabeled_arcs_ += num_actual_gold_arcs;
            task3_num_predicted_unlabeled_arcs_ += num_predicted_unlabeled_arcs;
            task3_num_matched_unlabeled_arcs_ += num_matched_unlabeled_arcs;

            task3_num_gold_labeled_arcs_ += num_actual_gold_arcs;
            task3_num_predicted_labeled_arcs_ += num_predicted_labeled_arcs;
            task3_num_matched_labeled_arcs_ += num_matched_labeled_arcs;

            task3_num_pruned_gold_unlabeled_arcs_ += missed_unlabeled;
            task3_num_pruned_gold_labeled_arcs_ += missed_labeled;
        }
    }

    virtual void EndEvaluation(double &all_unlabeled_F1, double &all_labeled_F1) {
        all_num_predicted_unlabeled_arcs_ =
                task1_num_predicted_unlabeled_arcs_ + task2_num_predicted_unlabeled_arcs_ + task3_num_predicted_unlabeled_arcs_;
        all_num_gold_unlabeled_arcs_ =
                task1_num_gold_unlabeled_arcs_ + task2_num_gold_unlabeled_arcs_ + task3_num_gold_unlabeled_arcs_;
        all_num_matched_unlabeled_arcs_ =
                task1_num_matched_unlabeled_arcs_ + task2_num_matched_unlabeled_arcs_ + task3_num_matched_unlabeled_arcs_;
        all_num_predicted_labeled_arcs_ =
                task1_num_predicted_labeled_arcs_ + task2_num_predicted_labeled_arcs_ + task3_num_predicted_labeled_arcs_;
        all_num_gold_labeled_arcs_ =
                task1_num_gold_labeled_arcs_ + task2_num_gold_labeled_arcs_ + task3_num_gold_labeled_arcs_;
        all_num_matched_labeled_arcs_ =
                task1_num_matched_labeled_arcs_ + task2_num_matched_labeled_arcs_ + task3_num_matched_labeled_arcs_;

        double task1_pruning_unlabeled_recall =
                static_cast<double>(task1_num_gold_unlabeled_arcs_ -
                                    task1_num_pruned_gold_unlabeled_arcs_) /
                static_cast<double>(task1_num_gold_unlabeled_arcs_);
        double task2_pruning_unlabeled_recall =
                static_cast<double>(task2_num_gold_unlabeled_arcs_ -
                                    task2_num_pruned_gold_unlabeled_arcs_) /
                static_cast<double>(task2_num_gold_unlabeled_arcs_);
        double task3_pruning_unlabeled_recall =
                static_cast<double>(task3_num_gold_unlabeled_arcs_ -
                                    task3_num_pruned_gold_unlabeled_arcs_) /
                static_cast<double>(task3_num_gold_unlabeled_arcs_);

        double task1_pruning_labeled_recall =
                static_cast<double>(task1_num_gold_labeled_arcs_ -
                                    task1_num_pruned_gold_labeled_arcs_) /
                static_cast<double>(task1_num_gold_labeled_arcs_);
        double task2_pruning_labeled_recall =
                static_cast<double>(task2_num_gold_labeled_arcs_ -
                                    task2_num_pruned_gold_labeled_arcs_) /
                static_cast<double>(task2_num_gold_labeled_arcs_);
        double task3_pruning_labeled_recall =
                static_cast<double>(task3_num_gold_labeled_arcs_ -
                                    task3_num_pruned_gold_labeled_arcs_) /
                static_cast<double>(task3_num_gold_labeled_arcs_);



        double task1_unlabeled_F1 = 0, task1_labeled_F1 = 0;
        double task1_unlabeled_precision =
                static_cast<double>(task1_num_matched_unlabeled_arcs_) /
                static_cast<double>(task1_num_predicted_unlabeled_arcs_);
        double task1_unlabeled_recall =
                static_cast<double>(task1_num_matched_unlabeled_arcs_) /
                static_cast<double>(task1_num_gold_unlabeled_arcs_);
        task1_unlabeled_F1 = 2.0 * task1_unlabeled_precision * task1_unlabeled_recall /
                          (task1_unlabeled_precision + task1_unlabeled_recall);

        double task1_labeled_precision =
                static_cast<double>(task1_num_matched_labeled_arcs_) /
                static_cast<double>(task1_num_predicted_labeled_arcs_);
        double task1_labeled_recall =
                static_cast<double>(task1_num_matched_labeled_arcs_) /
                static_cast<double>(task1_num_gold_labeled_arcs_);
        task1_labeled_F1 = 2.0 * task1_labeled_precision * task1_labeled_recall /
                        (task1_labeled_precision + task1_labeled_recall);
        double task2_unlabeled_F1 = 0, task2_labeled_F1 = 0;
        double task2_unlabeled_precision =
                static_cast<double>(task2_num_matched_unlabeled_arcs_) /
                static_cast<double>(task2_num_predicted_unlabeled_arcs_);
        double task2_unlabeled_recall =
                static_cast<double>(task2_num_matched_unlabeled_arcs_) /
                static_cast<double>(task2_num_gold_unlabeled_arcs_);
        task2_unlabeled_F1 = 2.0 * task2_unlabeled_precision * task2_unlabeled_recall /
                           (task2_unlabeled_precision + task2_unlabeled_recall);

        double task2_labeled_precision =
                static_cast<double>(task2_num_matched_labeled_arcs_) /
                static_cast<double>(task2_num_predicted_labeled_arcs_);
        double task2_labeled_recall =
                static_cast<double>(task2_num_matched_labeled_arcs_) /
                static_cast<double>(task2_num_gold_labeled_arcs_);
        task2_labeled_F1 = 2.0 * task2_labeled_precision * task2_labeled_recall /
                         (task2_labeled_precision + task2_labeled_recall);

        double task3_unlabeled_F1 = 0, task3_labeled_F1 = 0;
        double task3_unlabeled_precision =
                static_cast<double>(task3_num_matched_unlabeled_arcs_) /
                static_cast<double>(task3_num_predicted_unlabeled_arcs_);
        double task3_unlabeled_recall =
                static_cast<double>(task3_num_matched_unlabeled_arcs_) /
                static_cast<double>(task3_num_gold_unlabeled_arcs_);
        task3_unlabeled_F1 = 2.0 * task3_unlabeled_precision * task3_unlabeled_recall /
                           (task3_unlabeled_precision + task3_unlabeled_recall);

        double task3_labeled_precision =
                static_cast<double>(task3_num_matched_labeled_arcs_) /
                static_cast<double>(task3_num_predicted_labeled_arcs_);
        double task3_labeled_recall =
                static_cast<double>(task3_num_matched_labeled_arcs_) /
                static_cast<double>(task3_num_gold_labeled_arcs_);
        task3_labeled_F1 = 2.0 * task3_labeled_precision * task3_labeled_recall /
                         (task3_labeled_precision + task3_labeled_recall);


        double all_unlabeled_precision =
                static_cast<double>(all_num_matched_unlabeled_arcs_) /
                static_cast<double>(all_num_predicted_unlabeled_arcs_);
        double all_unlabeled_recall =
                static_cast<double>(all_num_matched_unlabeled_arcs_) /
                static_cast<double>(all_num_gold_unlabeled_arcs_);
        all_unlabeled_F1 = 2.0 * all_unlabeled_precision * all_unlabeled_recall /
                           (all_unlabeled_precision + all_unlabeled_recall);

        double all_labeled_precision =
                static_cast<double>(all_num_matched_labeled_arcs_) /
                static_cast<double>(all_num_predicted_labeled_arcs_);
        double all_labeled_recall =
                static_cast<double>(all_num_matched_labeled_arcs_) /
                static_cast<double>(all_num_gold_labeled_arcs_);
        all_labeled_F1 = 2.0 * all_labeled_precision * all_labeled_recall /
                         (all_labeled_precision + all_labeled_recall);


        LOG(INFO) << "DM UP: " << task1_unlabeled_precision
                  << " (" << task1_num_matched_unlabeled_arcs_ << "/"
                  << task1_num_predicted_unlabeled_arcs_ << ")"
                  << " UR: " << task1_unlabeled_recall
                  << " (" << task1_num_matched_unlabeled_arcs_ << "/"
                  << task1_num_gold_unlabeled_arcs_ << ")"
                  << " LP: " << task1_labeled_precision
                  << " (" << task1_num_matched_labeled_arcs_ << "/"
                  << task1_num_predicted_labeled_arcs_ << ")"
                  << " LR " << task1_labeled_recall
                  << " (" << task1_num_matched_labeled_arcs_ << "/"
                  << task1_num_gold_labeled_arcs_<< ")"
                  <<" UF: " << task1_unlabeled_F1
                  << " LF: " << task1_labeled_F1;

        LOG(INFO) << "PAS UP: " << task2_unlabeled_precision
                  << " (" << task2_num_matched_unlabeled_arcs_ << "/"
                  << task2_num_predicted_unlabeled_arcs_ << ")"
                  << " UR: " << task2_unlabeled_recall
                  << " (" << task2_num_matched_unlabeled_arcs_ << "/"
                  << task2_num_gold_unlabeled_arcs_ << ")"
                  << " LP: " << task2_labeled_precision
                  << " (" << task2_num_matched_labeled_arcs_ << "/"
                  << task2_num_predicted_labeled_arcs_ << ")"
                  << " LR " << task2_labeled_recall
                  << " (" << task2_num_matched_labeled_arcs_ << "/"
                  << task2_num_gold_labeled_arcs_ << ")"
                  << " UF: " << task2_unlabeled_F1
                  << " LF: " << task2_labeled_F1;

        LOG(INFO) << "PSD UP: " << task3_unlabeled_precision
                  << " (" << task3_num_matched_unlabeled_arcs_ << "/"
                  << task3_num_predicted_unlabeled_arcs_ << ")"
                  << " UR: " << task3_unlabeled_recall
                  << " (" << task3_num_matched_unlabeled_arcs_ << "/"
                  << task3_num_gold_unlabeled_arcs_ << ")"
                  << " LP: " << task3_labeled_precision
                  << " (" << task3_num_matched_labeled_arcs_ << "/"
                  << task3_num_predicted_labeled_arcs_ << ")"
                  << " LR " << task3_labeled_recall
                  << " (" << task3_num_matched_labeled_arcs_ << "/"
                  << task3_num_gold_labeled_arcs_<< ")"
                  <<" UF: " << task3_unlabeled_F1
                  << " LF: " << task3_labeled_F1;

        LOG(INFO)<< "Overall UP: " << all_unlabeled_precision
                          << " (" << all_num_matched_unlabeled_arcs_ << "/"
                          << all_num_predicted_unlabeled_arcs_ << ")"
                          << " UR: " << all_unlabeled_recall
                          << " (" << all_num_matched_unlabeled_arcs_ << "/"
                          << all_num_gold_unlabeled_arcs_ << ")"
                          << " LP: " << all_labeled_precision
                          << " (" << all_num_matched_labeled_arcs_ << "/"
                          << all_num_predicted_labeled_arcs_ << ")"
                          << " LR " << all_labeled_recall
                          << " (" << all_num_matched_labeled_arcs_ << "/"
                          << all_num_gold_labeled_arcs_<< ")"
                          <<" UF: " << all_unlabeled_F1
                          << " LF: " << all_labeled_F1;

//        LOG(INFO) << "DM Pruning unlabeled recall: " << task1_pruning_unlabeled_recall
//                  << " ("
//                  << task1_num_gold_unlabeled_arcs_ - task1_num_pruned_gold_unlabeled_arcs_
//                  << "/"
//                  << task1_num_gold_unlabeled_arcs_ << ")";
//        LOG(INFO) << "task2 Pruning unlabeled recall: " << task2_pruning_unlabeled_recall
//                  << " ("
//                  << task2_num_gold_unlabeled_arcs_ - task2_num_pruned_gold_unlabeled_arcs_
//                  << "/"
//                  << task2_num_gold_unlabeled_arcs_ << ")";
//        LOG(INFO) << "task3 Pruning unlabeled recall: " << task3_pruning_unlabeled_recall
//                  << " ("
//                  << task3_num_gold_unlabeled_arcs_ - task3_num_pruned_gold_unlabeled_arcs_
//                  << "/"
//                  << task3_num_gold_unlabeled_arcs_ << ")";
//
//
//        LOG(INFO) << "DM Pruning labeled recall: " << task1_pruning_labeled_recall
//                  << " ("
//                  << task1_num_gold_labeled_arcs_ - task1_num_pruned_gold_labeled_arcs_
//                  << "/"
//                  << task1_num_gold_labeled_arcs_ << ")";
//        LOG(INFO) << "task2 Pruning labeled recall: " << task2_pruning_labeled_recall
//                  << " ("
//                  << task2_num_gold_labeled_arcs_ - task2_num_pruned_gold_labeled_arcs_
//                  << "/"
//                  << task2_num_gold_labeled_arcs_ << ")";
//        LOG(INFO) << "task3 Pruning labeled recall: " << task3_pruning_labeled_recall
//                  << " ("
//                  << task3_num_gold_labeled_arcs_ - task3_num_pruned_gold_labeled_arcs_
//                  << "/"
//                  << task3_num_gold_labeled_arcs_ << ")";


        timeval end_clock;
        gettimeofday(&end_clock, NULL);
        double num_seconds =
                static_cast<double>(diff_ms(end_clock, start_clock_)) / 1000.0;
        double tokens_per_second = static_cast<double>(num_tokens_) / num_seconds;
//		LOG(INFO) << "Speed: "
//			<< tokens_per_second << " tokens per second.";
    }

#if 0
    void GetAllAncestors(const vector<int> &heads,
                                             int descend,
                                             vector<int>* ancestors);
    bool ExistsPath(const vector<int> &heads,
                                    int ancest,
                                    int descend);
#endif
public:
    dynet::Model *model;
    biLSTM *parser;
    dynet::Trainer *trainer;
protected:
    Reader *task1_reader_;
    Reader *task2_reader_;
    Reader *task3_reader_;

    Writer *task1_writer_;
    Writer *task2_writer_;
    Writer *task3_writer_;

    TokenDictionary *task1_token_dictionary_;
    TokenDictionary *task2_token_dictionary_;
    TokenDictionary *task3_token_dictionary_;

    //DependencyDictionary *dependency_dictionary_;
    DependencyDictionary *task1_dependency_dictionary_;
    DependencyDictionary *task2_dependency_dictionary_;
    DependencyDictionary *task3_dependency_dictionary_;

    Dictionary *task1_dictionary_;
    Dictionary *task2_dictionary_;
    Dictionary *task3_dictionary_;

    bool train_pruner_;
    Parameters *pruner_parameters_;
    Parameters *task1_pruner_parameters_;
    Parameters *task2_pruner_parameters_;
    Parameters *task3_pruner_parameters_;
    int num_tokens_;
    int num_unlabeled_arcs_after_pruning_;
    int num_labeled_arcs_after_pruning_;


    int all_num_predicted_unlabeled_arcs_;
    int all_num_gold_unlabeled_arcs_;
    int all_num_matched_unlabeled_arcs_;
    int all_num_gold_labeled_arcs_;
    int all_num_matched_labeled_arcs_;
    int all_num_predicted_labeled_arcs_;
    int all_num_pruned_gold_unlabeled_arcs_;
    int all_num_pruned_gold_labeled_arcs_;

    int task1_num_predicted_unlabeled_arcs_;
    int task1_num_gold_unlabeled_arcs_;
    int task1_num_matched_unlabeled_arcs_;
    int task1_num_gold_labeled_arcs_;
    int task1_num_matched_labeled_arcs_;
    int task1_num_predicted_labeled_arcs_;
    int task1_num_pruned_gold_unlabeled_arcs_;
    int task1_num_pruned_gold_labeled_arcs_;

    int task2_num_predicted_unlabeled_arcs_;
    int task2_num_gold_unlabeled_arcs_;
    int task2_num_matched_unlabeled_arcs_;
    int task2_num_gold_labeled_arcs_;
    int task2_num_matched_labeled_arcs_;
    int task2_num_predicted_labeled_arcs_;
    int task2_num_pruned_gold_unlabeled_arcs_;
    int task2_num_pruned_gold_labeled_arcs_;

    int task3_num_predicted_unlabeled_arcs_;
    int task3_num_gold_unlabeled_arcs_;
    int task3_num_matched_unlabeled_arcs_;
    int task3_num_gold_labeled_arcs_;
    int task3_num_matched_labeled_arcs_;
    int task3_num_predicted_labeled_arcs_;
    int task3_num_pruned_gold_unlabeled_arcs_;
    int task3_num_pruned_gold_labeled_arcs_;

    vector<Instance *> task1_instances_;
    vector<Instance *> task2_instances_;
    vector<Instance *> task3_instances_;
    timeval start_clock_;
    unordered_map<int, vector<float>> *embedding_;
    unordered_map<int, int> * form_count_;
    map <int, int> allowed_crossform_labeles_;
};

#endif /* NeuralSemanticPipe_H_ */
