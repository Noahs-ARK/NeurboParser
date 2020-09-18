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

#include "SemanticOptions.h"
#include "StringUtils.h"
#include "SerializationUtils.h"
#include <glog/logging.h>

using namespace std;

DEFINE_string(file_frames, "",
              "Path to the file containing the frames.");

DEFINE_string(file_exemplar, "",
              "Path to the file containing the exemplar sentences.");

DEFINE_bool(use_exemplar, true,
            "Use exemplar or not.");

DEFINE_double(exemplar_fraction, 0.0, "Fraction of exemplar instances to use.");

// TODO: Implement the text format.
DEFINE_string(srl_file_format, "conll",
              "Format of the input file containing the data. Use ""conll"" for "
		              "the format used in CONLL 2008, ""sdp"" for the format in "
		              "SemEval 2014, and ""text"" for tokenized sentences"
		              "(one per line, with tokens separated by white-spaces.");
DEFINE_string(srl_model_type, "standard",
              "Model type. This a string formed by the one or several of the "
		              "following pieces:"
		              "af enables arc-factored parts (required), "
		              "+as enables arbitrary sibling parts,"
		              "+cs enables consecutive sibling parts, "
		              "+gp enables grandparent parts,"
		              "+cp enables co-parent parts,"
		              "+ccp enables consecutive co-parent parts, "
		              "+gs enables grandsibling parts,"
		              "+ts enables trisibling parts,"
		              "+gs enables grand-sibling (third-order) parts,"
		              "+ts enables tri-sibling (third-order) parts."
		              "The following alias are predefined:"
		              "basic is af, "
		              "standard is af+cs+gp, "
		              "full is af+cs+gp+as+gs+ts.");
DEFINE_bool(srl_labeled, true,
            "True for training a parser with labeled arcs (if false, the "
		            "parser outputs just the backbone dependencies.)");
DEFINE_bool(srl_deterministic_labels, true,
            "True for forcing a set of labels (found in the training set) to be "
		            "deterministic (i.e. to not occur in more than one argument for the "
		            "same predicate).");
DEFINE_bool(srl_prune_basic, false,
            "True for using a basic pruner from a probabilistic first-order "
		            "model.");
DEFINE_string(file_pruner_model, "",
              "Path to the file containing the pre-trained pruner model. Must "
		              "activate the flag --use_pretrained_pruner");
DEFINE_double(srl_pruner_posterior_threshold, 0.0001,
              "Posterior probability threshold for an arc to be pruned, in basic "
		              "pruning. For each word p, if "
		              "P(p,a) < pruner_posterior_threshold * P(p,a'), "
		              "where a' is the best scored argument, then (p,a) will be pruned out.");

// Options for pruner training.
// TODO: implement these options.
DEFINE_int32(srl_pruner_train_epochs, 10,
             "Number of training epochs for the pruner.");
DEFINE_double(dropout_rate, 0.0, "Dropout rate.");
DEFINE_double(word_dropout_rate, 0.0, "Word dropout rate.");
DEFINE_bool(use_pretrained_embedding, false, "Optional use of pretrained embedding.");
DEFINE_string(file_pretrained_embedding, "path_to_embedding", "If using pretrained embedding, provide the path");
DEFINE_int32(num_lstm_layers, 2, "Number of layers of biLSTM encoder.");
DEFINE_int32(pruner_num_lstm_layers, 1, "Number of layers of biLSTM encoder.");
DEFINE_int32(lemma_dim, 25, "Dimension of pretrained word embedding.");
DEFINE_int32(word_dim, 100, "Dimension of word embedding.");
DEFINE_int32(pos_dim, 25, "Dimension of POS tag embedding.");
DEFINE_int32(lstm_dim, 200, "Dimension of biLSTM.");
DEFINE_int32(pruner_lstm_dim, 32, "Dimension of biLSTM.");
DEFINE_int32(mlp_dim, 100, "Dimension of MLP.");
DEFINE_int32(pruner_mlp_dim, 32, "Dimension of MLP.");
DEFINE_string(trainer, "adam", "Trainer to use: sgd_momentum, adam, adadelta");
DEFINE_string(output_term, "tensor", "Output term. tensor or conc");

DEFINE_bool(srl_train_pruner, false,
            "True if using a pre-trained basic pruner. Must specify the file "
		            "path through --file_pruner_model. If this flag is set to false "
		            "and train=true and prune_basic=true, a pruner will be trained "
		            "along with the parser.");
DEFINE_int32(max_span_length, 20, "FrameNet max span length.");
DEFINE_int32(max_dist, 20, "FrameNet max span length.");
DEFINE_uint64(num_updates, 0, "used for dealint with weight_decay in save/load.");
DEFINE_uint64(pruner_num_updates, 0, "used for dealint with weight_decay in save/load.");
DEFINE_double(eta0, 0.1, "eta0");
DEFINE_double(eta_decay, 0.05, "eta decay");
DEFINE_int32(halve, 0, "scheduled halving. set to 0 to disable");
DEFINE_int32(batch_size, 1, "");
DEFINE_bool(use_elmo, false, "");
DEFINE_string(file_elmo, "path_to_elmo_embedding", "");

// Save current option flags to the model file.
void SemanticOptions::Save(FILE *fs) {
	Options::Save(fs);

	bool success;
	success = WriteString(fs, model_type_);
	CHECK(success);
	success = WriteBool(fs, labeled_);
	CHECK(success);
	success = WriteBool(fs, deterministic_labels_);
	CHECK(success);
	success = WriteBool(fs, prune_basic_);
	CHECK(success);
	success = WriteDouble(fs, pruner_posterior_threshold_);
	CHECK(success);
	success = WriteDouble(fs, dropout_rate_);
	CHECK(success);
	success = WriteDouble(fs, word_dropout_rate_);
	CHECK(success);
	success = WriteBool(fs, use_pretrained_embedding_);
	CHECK(success);
	success = WriteString(fs, file_pretrained_embedding_);
	CHECK(success);
	success = WriteInteger(fs, num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, pruner_num_lstm_layers_);
	CHECK(success);
	success = WriteInteger(fs, lemma_dim_);
	CHECK(success);
	success = WriteInteger(fs, word_dim_);
	CHECK(success);
	success = WriteInteger(fs, pos_dim_);
	CHECK(success);
	success = WriteInteger(fs, lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, mlp_dim_);
	CHECK(success);
	success = WriteInteger(fs, pruner_lstm_dim_);
	CHECK(success);
	success = WriteInteger(fs, pruner_mlp_dim_);
	CHECK(success);
	success = WriteString(fs, trainer_);
	CHECK(success);
	success = WriteString(fs, output_term_);
	CHECK(success);
	success = WriteInteger(fs, max_span_length_);
	CHECK(success);
	success = WriteInteger(fs, max_dist_);
	CHECK(success);
	success = WriteUINT64(fs, num_updates_);
	CHECK(success);
	success = WriteUINT64(fs, pruner_num_updates_);
	CHECK(success);
	success = WriteBool(fs, use_elmo_);
	CHECK(success);
}

// Load current option flags to the model file.
// Note: this will override the user-specified flags.
void SemanticOptions::Load(FILE *fs) {
	Options::Load(fs);

	bool success;
	success = ReadString(fs, &FLAGS_srl_model_type);
	CHECK(success);
	LOG(INFO) << "Setting --srl_model_type=" << FLAGS_srl_model_type;

	success = ReadBool(fs, &FLAGS_srl_labeled);
	CHECK(success);
	LOG(INFO) << "Setting --srl_labeled=" << FLAGS_srl_labeled;

	success = ReadBool(fs, &FLAGS_srl_deterministic_labels);
	CHECK(success);
	LOG(INFO) << "Setting --srl_deterministic_labels=" << FLAGS_srl_deterministic_labels;

	success = ReadBool(fs, &FLAGS_srl_prune_basic);
	CHECK(success);
	LOG(INFO) << "Setting --srl_prune_basic=" << FLAGS_srl_prune_basic;

	success = ReadDouble(fs, &FLAGS_srl_pruner_posterior_threshold);
	CHECK(success);
	LOG(INFO) << "Setting --srl_pruner_posterior_threshold="
	          << FLAGS_srl_pruner_posterior_threshold;

	success = ReadDouble(fs, &FLAGS_dropout_rate);
	CHECK(success);
	LOG(INFO) << "Setting --dropout_rate="
	          << FLAGS_dropout_rate;

	success = ReadDouble(fs, &FLAGS_word_dropout_rate);
	CHECK(success);
	LOG(INFO) << "Setting --word_dropout_rate="
	          << FLAGS_word_dropout_rate;

	success = ReadBool(fs, &FLAGS_use_pretrained_embedding);
	CHECK(success);
	LOG(INFO) << "Setting --use_pretrained_embedding="
	          << FLAGS_use_pretrained_embedding;

	success = ReadString(fs, &FLAGS_file_pretrained_embedding);
	CHECK(success);
	LOG(INFO) << "Setting --file_pretrained_embedding="
	          << FLAGS_file_pretrained_embedding;

	success = ReadInteger(fs, &FLAGS_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --srl_num_lstm_layers="
	          << FLAGS_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_pruner_num_lstm_layers);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_num_lstm_layers="
	          << FLAGS_pruner_num_lstm_layers;

	success = ReadInteger(fs, &FLAGS_lemma_dim);
	CHECK(success);
	LOG(INFO) << "Setting --lemma_dim="
	          << FLAGS_lemma_dim;

	success = ReadInteger(fs, &FLAGS_word_dim);
	CHECK(success);
	LOG(INFO) << "Setting --word_dim="
	          << FLAGS_word_dim;

	success = ReadInteger(fs, &FLAGS_pos_dim);
	CHECK(success);
	LOG(INFO) << "Setting --pos_dim="
	          << FLAGS_pos_dim;

	success = ReadInteger(fs, &FLAGS_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --lstm_dim="
	          << FLAGS_lstm_dim;

	success = ReadInteger(fs, &FLAGS_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --mlp_dim="
	          << FLAGS_mlp_dim;

	success = ReadInteger(fs, &FLAGS_pruner_lstm_dim);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_lstm_dim="
	          << FLAGS_pruner_lstm_dim;

	success = ReadInteger(fs, &FLAGS_pruner_mlp_dim);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_mlp_dim="
	          << FLAGS_pruner_mlp_dim;

	success = ReadString(fs, &FLAGS_trainer);
	CHECK(success);
	LOG(INFO) << "Setting --trainer="
	          << FLAGS_trainer;

	success = ReadString(fs, &FLAGS_output_term);
	CHECK(success);
	LOG(INFO) << "Setting --output_term="
	          << FLAGS_output_term;

	success = ReadInteger(fs, &FLAGS_max_span_length);
	CHECK(success);
	LOG(INFO) << "Setting --max_span_length="
	          << FLAGS_max_span_length;

	success = ReadInteger(fs, &FLAGS_max_dist);
	CHECK(success);
	LOG(INFO) << "Setting --max_dist="
	          << FLAGS_max_dist;

	success = ReadUINT64(fs, &FLAGS_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --num_updates="
	          << FLAGS_num_updates;

	success = ReadUINT64(fs, &FLAGS_pruner_num_updates);
	CHECK(success);
	LOG(INFO) << "Setting --pruner_num_updates="
	          << FLAGS_pruner_num_updates;

	success = ReadBool(fs, &FLAGS_use_elmo);
	CHECK(success);
	LOG(INFO) << "Setting --use_elmo="
	          << FLAGS_use_elmo;
	Initialize();
}

void SemanticOptions::Initialize() {
	Options::Initialize();
	file_frames_ = FLAGS_file_frames;
	file_exemplar_ = FLAGS_file_exemplar;
	use_exemplar_ = FLAGS_use_exemplar;
	exemplar_fraction_ = FLAGS_exemplar_fraction;
	file_format_ = FLAGS_srl_file_format;
	model_type_ = FLAGS_srl_model_type;
	labeled_ = FLAGS_srl_labeled;
	deterministic_labels_ = FLAGS_srl_deterministic_labels;
	prune_basic_ = FLAGS_srl_prune_basic;
	file_pruner_model_ = FLAGS_file_pruner_model;
	pruner_posterior_threshold_ = FLAGS_srl_pruner_posterior_threshold;
	dropout_rate_ = FLAGS_dropout_rate;
	word_dropout_rate_ = FLAGS_word_dropout_rate;
	use_pretrained_embedding_ = FLAGS_use_pretrained_embedding;
	file_pretrained_embedding_ = FLAGS_file_pretrained_embedding;
	num_lstm_layers_ = FLAGS_num_lstm_layers;
	pruner_num_lstm_layers_ = FLAGS_pruner_num_lstm_layers;
	lemma_dim_ = FLAGS_lemma_dim;
	word_dim_ = FLAGS_word_dim;
	pos_dim_ = FLAGS_pos_dim;
	lstm_dim_ = FLAGS_lstm_dim;
	pruner_lstm_dim_ = FLAGS_pruner_lstm_dim;
	mlp_dim_ = FLAGS_mlp_dim;
	pruner_mlp_dim_ = FLAGS_pruner_mlp_dim;
	trainer_ = FLAGS_trainer;
	output_term_ = FLAGS_output_term;
	train_pruner_ = FLAGS_srl_train_pruner;
	max_span_length_ = FLAGS_max_span_length;
	max_dist_ = FLAGS_max_dist;
	num_updates_ = FLAGS_num_updates;
	pruner_num_updates_ = FLAGS_pruner_num_updates;
	eta0_ = FLAGS_eta0;
	eta_decay_ = FLAGS_eta_decay;
	halve_ = FLAGS_halve;
	batch_size_ = FLAGS_batch_size;
	use_elmo_ = FLAGS_use_elmo;
	file_elmo_ = FLAGS_file_elmo;

	// Enable the parts corresponding to the model type.
	string model_type = FLAGS_srl_model_type;
	if (model_type == "basic") {
		model_type = "af";
	}
	vector<string> enabled_parts;
	bool use_arc_factored = false;
	StringSplit(model_type, "+", &enabled_parts, true);
	for (int i = 0; i < enabled_parts.size(); ++i) {
		if (enabled_parts[i] == "af") {
			use_arc_factored = true;
			LOG(INFO) << "Arc factored parts enabled.";
		} else {
			CHECK(false) << "Unknown part in model type: " << enabled_parts[i];
		}
	}

	CHECK(use_arc_factored) << "Arc-factored parts are mandatory in model type";
}
