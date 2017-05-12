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

class NeuralSemanticPipe : public Pipe {
public:
    NeuralSemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
        token_dictionary_ = NULL;
        dependency_dictionary_ = NULL;
        pruner_parameters_ = NULL;
        train_pruner_ = false;
        trainer = NULL;
        model = NULL;
        parser = NULL;
    }

    virtual ~NeuralSemanticPipe() {
        delete token_dictionary_;
        delete dependency_dictionary_;
        delete pruner_parameters_;
        delete trainer;
        delete parser;
        delete model;
    }

    SemanticReader *GetSemanticReader() {
        return static_cast<SemanticReader *>(reader_);
    }

    SemanticDictionary *GetSemanticDictionary() {
        return static_cast<SemanticDictionary *>(dictionary_);
    }

    SemanticDecoder *GetSemanticDecoder() {
        return static_cast<SemanticDecoder *>(decoder_);
    }

    SemanticOptions *GetSemanticOptions() {
        return static_cast<SemanticOptions *>(options_);
    }

    void Initialize() {
        Pipe::Initialize();
        pruner_parameters_ = new Parameters;
    }

    void NeuralInitialize() {
        model = new dynet::Model();
        SemanticOptions *semantic_options = GetSemanticOptions();

        if (semantic_options->trainer() == "adadelta")
            trainer = new dynet::AdadeltaTrainer(*model);
        else if (semantic_options->trainer() == "adam") {
            trainer = new dynet::AdamTrainer(*model, 0.001, 0.9, 0.9, 1e-8);
        }
        else if (semantic_options->trainer() == "sgd") {
            trainer = new dynet::SimpleSGDTrainer(*model);
            trainer->eta_decay = 0.02;
        } else {
            LOG(INFO) << "Unsupported trainer. Giving up..." << endl;
            CHECK(false);
        }
        trainer->clip_threshold = 1.0;
        int num_roles = GetSemanticDictionary()->GetNumRoles();
        parser = new biLSTMParser<dynet::LSTMBuilder>(semantic_options, num_roles, model);
        parser->InitParams(model);
    }

    void SetPrunerParameters(Parameters *pruner_parameters) {
        pruner_parameters_ = pruner_parameters;
    }

    void LoadPretrainedEmbedding();

    void NeuralTrain();

    double NeuralTrainEpoch(const vector<int> &idxs, int epoch);

    void NeuralTest();

    void NeuralRun(double &unlabeled_F1, double &labeled_F1);

    void LoadNueralModel();

    void SaveNueralModel();

    void LoadPruner(const std::string &file_name) {
        FILE *fs = fopen(file_name.c_str(), "rb");
        CHECK(fs) << "Could not open model file for reading: " << file_name;
        pruner_parameters_->Load(fs);
        fclose(fs);
    }

    void SavePruner(const std::string &file_name) {
        FILE *fs = fopen(file_name.c_str(), "wb");
        CHECK(fs) << "Could not open model file for writing: " << file_name;
        pruner_parameters_->Save(fs);
        fclose(fs);

    }
protected:
    void CreateDictionary() {
        dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary()->SetTokenDictionary(token_dictionary_);
        GetSemanticDictionary()->SetDependencyDictionary(dependency_dictionary_);
    }

    void CreateReader() {
        reader_ = new SemanticReader(options_);
    }

    void CreateWriter() {
        writer_ = new SemanticWriter(options_);
    }

    void CreateDecoder() { decoder_ = new SemanticDecoder(this); }

    Parts *CreateParts() { return new SemanticParts; }

    Features *CreateFeatures() { return new SemanticFeatures(this); }

    void CreateTokenDictionary() {
        token_dictionary_ = new TokenDictionary(this);
    }

    void CreateDependencyDictionary() {
        dependency_dictionary_ = new DependencyDictionary(this);
    }

    Parameters *GetTrainingParameters() {
        if (train_pruner_) return pruner_parameters_;
        return parameters_;
    }

    void PreprocessData();

    Instance *GetFormattedInstance(Instance *instance) {
        SemanticInstanceNumeric *instance_numeric =
                new SemanticInstanceNumeric;
        instance_numeric->Initialize(*GetSemanticDictionary(), static_cast<SemanticInstance *>(instance));
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
        num_predicted_unlabeled_arcs_ = 0;
        num_gold_unlabeled_arcs_ = 0;
        num_matched_unlabeled_arcs_ = 0;
        num_tokens_ = 0;
        num_unlabeled_arcs_after_pruning_ = 0;
        num_pruned_gold_unlabeled_arcs_ = 0;
        num_possible_unlabeled_arcs_ = 0;
        num_predicted_labeled_arcs_ = 0;
        num_gold_labeled_arcs_ = 0;
        num_matched_labeled_arcs_ = 0;
        num_labeled_arcs_after_pruning_ = 0;
        num_pruned_gold_labeled_arcs_ = 0;
        num_possible_labeled_arcs_ = 0;
        gettimeofday(&start_clock_, NULL);
    }

    virtual void EvaluateInstance(Instance *instance, Instance *output_instance,
                                  Parts *parts, const vector<double> &gold_outputs,
                                  const vector<double> &predicted_outputs) {
        int num_possible_unlabeled_arcs = 0;
        int num_possible_labeled_arcs = 0;
        int num_gold_unlabeled_arcs = 0;
        int num_gold_labeled_arcs = 0;
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
                            ++num_matched_unlabeled_arcs_;
                        }
                        ++num_gold_unlabeled_arcs;
                    }
                    if (predicted_outputs[r] >= 0.5) {
                        CHECK_EQ(predicted_outputs[r], 1.0);
                        ++num_predicted_unlabeled_arcs_;

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
                                    ++num_matched_labeled_arcs_;

                                    //LOG(INFO) << semantic_instance->GetForm(a)
                                    //          << " <-*- "
                                    //          << semantic_instance->GetForm(p);
                                }
                                ++num_gold_labeled_arcs;
                            }
                            if (predicted_outputs[r] >= 0.5) {
                                CHECK_EQ(predicted_outputs[r], 1.0);
                                ++num_predicted_labeled_arcs_;
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
        num_gold_unlabeled_arcs_ += num_actual_gold_arcs;
        num_gold_labeled_arcs_ += num_actual_gold_arcs;
        int missed_unlabeled = num_actual_gold_arcs - num_gold_unlabeled_arcs;
        int missed_labeled = num_actual_gold_arcs - num_gold_labeled_arcs;
        int missed = missed_unlabeled + missed_labeled;
//		if (missed > 0) {
//		  LOG(INFO) << "Missed " << missed_labeled << " labeled arcs.";
//		}
        num_pruned_gold_unlabeled_arcs_ += missed_unlabeled;
        num_possible_unlabeled_arcs_ += num_possible_unlabeled_arcs;
        num_pruned_gold_labeled_arcs_ += missed_labeled;
        num_possible_labeled_arcs_ += num_possible_labeled_arcs;
    }

    virtual void EndEvaluation(double &unlabeled_F1, double &labeled_F1) {
        double unlabeled_precision =
                static_cast<double>(num_matched_unlabeled_arcs_) /
                static_cast<double>(num_predicted_unlabeled_arcs_);
        double unlabeled_recall =
                static_cast<double>(num_matched_unlabeled_arcs_) /
                static_cast<double>(num_gold_unlabeled_arcs_);
        unlabeled_F1 = 2.0 * unlabeled_precision * unlabeled_recall /
                       (unlabeled_precision + unlabeled_recall);
        double pruning_unlabeled_recall =
                static_cast<double>(num_gold_unlabeled_arcs_ -
                                    num_pruned_gold_unlabeled_arcs_) /
                static_cast<double>(num_gold_unlabeled_arcs_);
        double pruning_unlabeled_efficiency =
                static_cast<double>(num_possible_unlabeled_arcs_) /
                static_cast<double>(num_tokens_);

        double labeled_precision =
                static_cast<double>(num_matched_labeled_arcs_) /
                static_cast<double>(num_predicted_labeled_arcs_);
        double labeled_recall =
                static_cast<double>(num_matched_labeled_arcs_) /
                static_cast<double>(num_gold_labeled_arcs_);
        labeled_F1 = 2.0 * labeled_precision * labeled_recall /
                     (labeled_precision + labeled_recall);
        double pruning_labeled_recall =
                static_cast<double>(num_gold_labeled_arcs_ -
                                    num_pruned_gold_labeled_arcs_) /
                static_cast<double>(num_gold_labeled_arcs_);
        double pruning_labeled_efficiency =
                static_cast<double>(num_possible_labeled_arcs_) /
                static_cast<double>(num_tokens_);

        LOG(INFO) << "Unlabeled precision: " << unlabeled_precision
                  << " (" << num_matched_unlabeled_arcs_ << "/"
                  << num_predicted_unlabeled_arcs_ << ")" << " recall: " << unlabeled_recall
                  << " (" << num_matched_unlabeled_arcs_ << "/"
                  << num_gold_unlabeled_arcs_ << ")" << " F1: " << unlabeled_F1;
//		LOG(INFO) << "Pruning unlabeled recall: " << pruning_unlabeled_recall
//			<< " ("
//			<< num_gold_unlabeled_arcs_ - num_pruned_gold_unlabeled_arcs_
//			<< "/"
//			<< num_gold_unlabeled_arcs_ << ")";
//		LOG(INFO) << "Pruning unlabeled efficiency: " << pruning_unlabeled_efficiency
//			<< " possible unlabeled arcs per token"
//			<< " (" << num_possible_unlabeled_arcs_ << "/"
//			<< num_tokens_ << ")";

        LOG(INFO) << "Labeled precision: " << labeled_precision
                  << " (" << num_matched_labeled_arcs_ << "/"
                  << num_predicted_labeled_arcs_ << ")" << " recall: " << labeled_recall
                  << " (" << num_matched_labeled_arcs_ << "/"
                  << num_gold_labeled_arcs_ << ")" << " F1: " << labeled_F1;
//		LOG(INFO) << "Pruning labeled recall: " << pruning_labeled_recall
//			<< " ("
//			<< num_gold_labeled_arcs_ - num_pruned_gold_labeled_arcs_
//			<< "/"
//			<< num_gold_labeled_arcs_ << ")";
//		LOG(INFO) << "Pruning labeled efficiency: " << pruning_labeled_efficiency
//			<< " possible labeled arcs per token"
//			<< " (" << num_possible_labeled_arcs_ << "/"
//			<< num_tokens_ << ")";

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
    TokenDictionary *token_dictionary_;
    DependencyDictionary *dependency_dictionary_;
    bool train_pruner_;
    Parameters *pruner_parameters_;
    int num_predicted_unlabeled_arcs_;
    int num_gold_unlabeled_arcs_;
    int num_matched_unlabeled_arcs_;
    int num_tokens_;
    int num_unlabeled_arcs_after_pruning_;
    int num_pruned_gold_unlabeled_arcs_;
    int num_possible_unlabeled_arcs_;
    int num_predicted_labeled_arcs_;
    int num_gold_labeled_arcs_;
    int num_matched_labeled_arcs_;
    int num_labeled_arcs_after_pruning_;
    int num_pruned_gold_labeled_arcs_;
    int num_possible_labeled_arcs_;
    timeval start_clock_;
    unordered_map<int, vector<float>> *embedding_;
    unordered_map<int, int> *form_count_;
};

#endif /* NeuralSemanticPipe_H_ */
