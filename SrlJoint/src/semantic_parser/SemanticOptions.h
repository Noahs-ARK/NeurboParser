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

    // Replace the flags by the pruner flags. This will overwrite some
    // of the flags. This function is called when training the pruner
    // along with the parser (rather than using an external pruner).
    void CopyPrunerFlags();

    // Get option values.
    const string &file_format() { return file_format_; }

    bool labeled() { return labeled_; }

    bool deterministic_labels() { return deterministic_labels_; }

    bool prune_basic() { return prune_basic_; }

    const string &GetPrunerModelFilePath() { return file_pruner_model_; }

    double GetPrunerPosteriorThreshold() { return pruner_posterior_threshold_; }

    void train_off() { train_ = false; }

    void train_on() { train_ = true; }

	float dropout_rate() { return dropout_rate_; }

    float word_dropout_rate() { return word_dropout_rate_; }

    bool use_pretrained_embedding() { return use_pretrained_embedding_; }

    const string &GetPretrainedEmbeddingFilePath() { return file_pretrained_embedding_; }

    int num_lstm_layers() { return num_lstm_layers_; }

    int pruner_num_lstm_layers() { return pruner_num_lstm_layers_; }

    int word_dim() { return word_dim_; }

    int lemma_dim() { return lemma_dim_; }

    int pos_dim() { return pos_dim_; }

    int lstm_dim() { return lstm_dim_; }

    int pruner_lstm_dim() { return pruner_lstm_dim_; }

    int mlp_dim() { return mlp_dim_; }

    int pruner_mlp_dim() { return pruner_mlp_dim_; }

    string trainer() { return trainer_; }

    const bool train_pruner() {
        return train_pruner_;
    }

    const string &GetFrameFilePath() {
        return file_frames_;
    }

    const string &GetExemplarFilePath() {
        return file_exemplar_;
    }

    const string &GetOutputTerm() {
        return output_term_;
    }

    void train_pruner_off() { train_pruner_ = false; }

    int pruner_epochs() { return train_pruner_epochs_; }

    int max_span_length() { return max_span_length_; }

    int max_dist() { return max_dist_; }

    bool use_exemplar() { return use_exemplar_; }

    float exemplar_fraction() { return  exemplar_fraction_; }

	float eta0() { return eta0_; }

	float eta_decay() { return eta_decay_; }

	int halve() { return halve_; }

	int batch_size() { return batch_size_; }

	bool use_elmo() { return use_elmo_; }

	const string &file_elmo() { return file_elmo_; }

    uint64_t num_updates_; // used for dealint with weight_decay in save/load.
    uint64_t pruner_num_updates_;
	float eta0_, eta_decay_;
protected:
    string file_format_;
    string model_type_;
    string file_frames_;
    string file_exemplar_;
    bool labeled_;
    bool deterministic_labels_;
    bool prune_basic_;
    bool train_pruner_;
    string file_pruner_model_;
    double pruner_posterior_threshold_;
    float dropout_rate_;
    float word_dropout_rate_;
    bool use_pretrained_embedding_;
    string file_pretrained_embedding_;
    int pruner_num_lstm_layers_;
    int num_lstm_layers_;
    int word_dim_;
    int lemma_dim_;
    int pos_dim_;
    int lstm_dim_, pruner_lstm_dim_;
    int mlp_dim_, pruner_mlp_dim_;
    string trainer_;
    string output_term_;
    int max_span_length_;
    int max_dist_;
    int train_pruner_epochs_;
    bool use_exemplar_;
    float exemplar_fraction_;
	int halve_;
	int batch_size_;
	bool use_elmo_;
	string file_elmo_;
};

#endif // SEMANTIC_OPTIONS_H_
