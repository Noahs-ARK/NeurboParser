//
// Created by hpeng on 7/7/17.
//

#ifndef SRL_BILSTM_H
#define SRL_BILSTM_H

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "SemanticPart.h"
#include "SemanticDecoder.h"

using namespace std;
using namespace dynet;
const int UNK_ID = 0;
const unsigned VOCAB_SIZE = 120000;
const unsigned POS_SIZE = 50;
const unsigned LEMMA_SIZE = 100000;
const unsigned LU_SIZE = 10000;
const unsigned FRAME_SIZE = 1500;

class BiLSTM {

protected:

	unsigned WORD_DIM;
	unsigned LEMMA_DIM;
	unsigned POS_DIM;
	unsigned LSTM_DIM;
	unsigned MLP_DIM;
	float DROPOUT;
	float WORD_DROPOUT;

	LSTMBuilder l2rbuilder_;
	LSTMBuilder r2lbuilder_;

	SemanticDecoder *decoder_;

	unordered_map<string, LookupParameter> lookup_params_;

public:

	explicit BiLSTM() {}

	explicit BiLSTM(int num_layers, int input_dim, int lstm_dim,
	                SemanticDecoder *decoder, ParameterCollection *model) :
			l2rbuilder_(num_layers, input_dim, lstm_dim, *model),
			r2lbuilder_(num_layers, input_dim, lstm_dim, *model),
			decoder_(decoder) {}

	virtual void InitParams(ParameterCollection *model) {}

	virtual Expression BuildGraph(Instance *instance,
	                              Parts *parts, vector<double> *scores,
	                              const vector<double> *gold_outputs,
	                              vector<double> *predicted_outputs,
	                              unordered_map<int, int> *form_count,
	                              bool is_train, ComputationGraph &cg) {}

	void LoadEmbedding(unordered_map<int, vector<float> > *Embedding) {
		for (auto it : (*Embedding)) {
			lookup_params_["embed_word_"].initialize(it.first, it.second);
		}
	}
};

#endif //SRL_BILSTM_H