//
// Created by hpeng on 11/3/16.
//

#ifndef NEURBOPARSER_BILSTM_H
#define NEURBOPARSER_BILSTM_H

#endif //NEURBOPARSER_BILSTM_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "SemanticOptions.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"
#include "SemanticPipe.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tr1/unordered_map>

using namespace std;
using namespace dynet;
const int UNK_ID = 0;
const unsigned VOCAB_SIZE = 35000;
const unsigned POS_SIZE = 49;

class BiLSTM {

protected:
    LookupParameter embed_word_;
    LookupParameter embed_lemma_;
    LookupParameter embed_pos_;

    unsigned WORD_DIM;
    unsigned LEMMA_DIM;
    unsigned POS_DIM;
    unsigned LSTM_DIM;
    unsigned MLP_DIM;
    unsigned LABEL_SIZE;

    LSTMBuilder l2rbuilder_;
    LSTMBuilder r2lbuilder_;

    Decoder *decoder_;

public:

    explicit BiLSTM() {}

    explicit BiLSTM(int num_layers, int input_dim, int lstm_dim, Decoder *decoder, ParameterCollection *model):
             l2rbuilder_(num_layers, input_dim, lstm_dim, *model),
             r2lbuilder_(num_layers, input_dim, lstm_dim, *model), decoder_(decoder) {

    }

    virtual void InitParams(ParameterCollection *model) {}

    virtual Expression BuildGraph(Instance *instance, Parts *parts, vector<double> &scores,
                                  const vector<double> &gold_outputs, vector<double> &predicted_outputs,
                                  const bool &use_word_dropout, const float &word_dropout_rate,
                                  unordered_map<int, int> *form_count,
                                  const bool &is_train, ComputationGraph &cg) {}

    void LoadEmbedding(unordered_map<int, vector<float> > *Embedding) {
        for (auto it : (*Embedding)) {
            embed_word_.initialize(it.first, it.second);
        }
    }
};