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

#ifndef SEMANTIC_OPTIONS_H_
#define SEMANTIC_OPTIONS_H_

#include "Options.h"

class SemanticOptions : public Options {
public:
    SemanticOptions() {};

    virtual ~SemanticOptions() {};

    // Serialization functions.
    void Load(FILE *fs);

    void Save(FILE *fs);

    // Initialization: set options based on the flags.
    void Initialize();

    // Get option values.
    const string &file_format() { return file_format_; }

    bool labeled() { return labeled_; }

    bool deterministic_labels() { return deterministic_labels_; }

    bool use_dependency_syntactic_features() {
        return use_dependency_syntactic_features_;
    }

    bool allow_self_loops() { return allow_self_loops_; }

    bool allow_root_predicate() { return allow_root_predicate_; }

    bool allow_unseen_predicates() { return allow_unseen_predicates_; }

    bool use_predicate_senses() { return use_predicate_senses_; }

    bool prune_labels() { return prune_labels_; }

    bool prune_labels_with_senses() { return prune_labels_with_senses_; }

    bool prune_labels_with_relation_paths() {
        return prune_labels_with_relation_paths_;
    }

    bool prune_distances() { return prune_distances_; }

    bool prune_basic() { return prune_basic_; }

    const string &GetPrunerModelFilePath() { return file_pruner_model_; }

    double GetPrunerPosteriorThreshold() { return pruner_posterior_threshold_; }

    double GetPrunerMaxArguments() { return pruner_max_arguments_; }

    void train_off() { train_ = false; }

    void train_on() { train_ = true; }

    bool use_word_dropout() { return use_word_dropout_; }

    float word_dropout_rate() { return word_dropout_rate_; }

    bool use_pretrained_embedding() { return use_pretrained_embedding_; }

    const string &GetPretrainedEmbeddingFilePath() { return file_pretrained_embedding_; }

    int num_lstm_layers() { return num_lstm_layers_; }

    int pruner_num_lstm_layers() { return pruner_num_lstm_layers_; }

    int lemma_dim() { return lemma_dim_; }

    int word_dim() { return word_dim_; }

    int pos_dim() { return pos_dim_; }

    int lstm_dim() { return lstm_dim_; }

    int pruner_lstm_dim() { return pruner_lstm_dim_; }

    int mlp_dim() { return mlp_dim_; }

    int pruner_mlp_dim() { return pruner_mlp_dim_; }

    string trainer() { return trainer_; }

    bool train_pruner() { return train_pruner_; }

    int pruner_epochs() { return pruner_epochs_; }

    void train_pruner_off() { train_pruner_ = false; }

    // temporary solution to weight_decay issue in dynet
    // TODO: save the weight_decay along with the model.
    uint64_t num_updates_;
    uint64_t pruner_num_updates_;

protected:
    string file_format_;
    string model_type_;
    bool use_dependency_syntactic_features_;
    bool labeled_;
    bool deterministic_labels_;
    bool allow_self_loops_;
    bool allow_root_predicate_;
    bool allow_unseen_predicates_;
    bool use_predicate_senses_;
    bool prune_labels_;
    bool prune_labels_with_senses_;
    bool prune_labels_with_relation_paths_;
    bool prune_distances_;
    bool prune_basic_;
    string file_pruner_model_;
    double pruner_posterior_threshold_;
    int pruner_max_arguments_;
    bool use_word_dropout_;
    float word_dropout_rate_;
    bool use_pretrained_embedding_;
    string file_pretrained_embedding_;
    int lemma_dim_;
    int word_dim_;
    int pos_dim_;
    int lstm_dim_, pruner_lstm_dim_;
    int mlp_dim_, pruner_mlp_dim_;
    int num_lstm_layers_, pruner_num_lstm_layers_;
    string trainer_;
    bool train_pruner_;
    int pruner_epochs_;
};

#endif // SEMANTIC_OPTIONS_H_
