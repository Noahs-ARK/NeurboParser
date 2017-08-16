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

#ifndef SemanticPipe_H_
#define SemanticPipe_H_

#include "Pipe.h"
#include "SemanticOptions.h"
#include "SemanticReader.h"
#include "SemanticDictionary.h"
#include "TokenDictionary.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticWriter.h"
#include "SemanticPart.h"
#include "SemanticDecoder.h"
#include "Parser.h"
#include "Pruner.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/io.h"

class SemanticPipe : public Pipe {
public:
    SemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
        token_dictionary_ = NULL;
        dependency_dictionary_ = NULL;
        trainer_ = NULL;
        model_ = NULL;
        parser_ = NULL;
        pruner_trainer_ = NULL;
        pruner_model_ = NULL;
        pruner_ = NULL;
    }

    virtual ~SemanticPipe() {
        delete token_dictionary_;
        delete dependency_dictionary_;
        delete trainer_;
        delete parser_;
        delete model_;
        delete pruner_trainer_;
        delete pruner_;
        delete pruner_model_;
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
        PreprocessData();
        model_ = new ParameterCollection();
        pruner_model_ = new ParameterCollection();
        SemanticOptions *semantic_options = GetSemanticOptions();

        if (semantic_options->trainer() == "adadelta") {
            trainer_ = new AdadeltaTrainer(*model_);
            pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
        } else if (semantic_options->trainer() == "adam") {
            trainer_ = new AdamTrainer(*model_, 0.001, 0.9, 0.9, 1e-8);
            pruner_trainer_ = new AdamTrainer(*pruner_model_, 0.001, 0.9, 0.9, 1e-8);
        } else if (semantic_options->trainer() == "sgd") {
            trainer_ = new SimpleSGDTrainer(*model_);
            pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_);
        } else {
            CHECK(false)<< "Unsupported trainer. Giving up..." << endl;
        }
        trainer_->clip_threshold = 1.0;
        pruner_trainer_->clip_threshold = 1.0;
        int num_roles = GetSemanticDictionary()->GetNumRoles();
        parser_ = new Parser(semantic_options, num_roles, decoder_, model_);
        parser_->InitParams(model_);
        pruner_ = new Pruner(semantic_options, num_roles, decoder_, pruner_model_);
        pruner_->InitParams(pruner_model_);
    }

    void LoadPretrainedEmbedding(bool load_parser_embedding, bool load_pruner_embedding);

    void BuildFormCount();

    void Train();

    void TrainPruner();

    double TrainEpoch(const vector<int> &idxs, int epoch);

    double TrainPrunerEpoch(const vector<int> &idxs, int epoch);

    void Test();

    void Run(double &unlabeled_F1, double &labeled_F1);

    void LoadNeuralModel();

    void SaveNeuralModel();

    void LoadPruner();

    void SavePruner();

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

    void CreateTokenDictionary() {
        token_dictionary_ = new TokenDictionary(this);
    }

    void CreateDependencyDictionary() {
        dependency_dictionary_ = new DependencyDictionary(this);
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
		LOG(INFO) << "Pruning unlabeled recall: " << pruning_unlabeled_recall
			<< " ("
			<< num_gold_unlabeled_arcs_ - num_pruned_gold_unlabeled_arcs_
			<< "/"
			<< num_gold_unlabeled_arcs_ << ")";
//		LOG(INFO) << "Pruning unlabeled efficiency: " << pruning_unlabeled_efficiency
//			<< " possible unlabeled arcs per token"
//			<< " (" << num_possible_unlabeled_arcs_ << "/"
//			<< num_tokens_ << ")";

        LOG(INFO) << "Labeled precision: " << labeled_precision
                  << " (" << num_matched_labeled_arcs_ << "/"
                  << num_predicted_labeled_arcs_ << ")" << " recall: " << labeled_recall
                  << " (" << num_matched_labeled_arcs_ << "/"
                  << num_gold_labeled_arcs_ << ")" << " F1: " << labeled_F1;
		LOG(INFO) << "Pruning labeled recall: " << pruning_labeled_recall
			<< " ("
			<< num_gold_labeled_arcs_ - num_pruned_gold_labeled_arcs_
			<< "/"
			<< num_gold_labeled_arcs_ << ")";
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

    /* Virtual function from Pipe.h but not implemented. */
    void ComputeScores(Instance *instance, Parts *parts, Features *features,
                       vector<double> *scores) {
        CHECK(false) << "Not implemented." << endl;
    }


    void RemoveUnsupportedFeatures(Instance *instance, Parts *parts,
                                   const vector<bool> &selected_parts,
                                   Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeFeatureDifference(Parts *parts,
                               Features *features,
                               const vector<double> &gold_output,
                               const vector<double> &predicted_output,
                               FeatureVector *difference) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeGradientStep(Parts *parts,
                          Features *features,
                          double eta,
                          int iteration,
                          const vector<double> &gold_output,
                          const vector<double> &predicted_output) {
        CHECK(false) << "Not implemented." << endl;
    }

    void TouchParameters(Parts *parts, Features *features,
                         const vector<bool> &selected_parts) {
        CHECK(false) << "Not implemented." << endl;
    }

    Features *CreateFeatures() { CHECK(false) << "Not implemented." << endl; }

    void MakeSelectedFeatures(Instance *instance, Parts *parts,
                              const vector<bool> &selected_parts, Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }

public:
    ParameterCollection *model_;
    ParameterCollection *pruner_model_;
    BiLSTM *parser_;
    BiLSTM *pruner_;
    Trainer *trainer_;
    Trainer *pruner_trainer_;

protected:
    TokenDictionary *token_dictionary_;
    DependencyDictionary *dependency_dictionary_;
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

#endif /* SemanticPipe_H_ */
