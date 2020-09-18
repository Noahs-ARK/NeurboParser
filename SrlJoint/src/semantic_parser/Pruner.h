//
// Created by hpeng on 7/7/17.
//

#ifndef SRL_PRUNER_H
#define SRL_PRUNER_H

#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SemanticPart.h"
#include "SemanticInstance.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"
#include "AlgUtils.h"

class Pruner : public BiLSTM {

protected:

    unsigned RANK;
    unsigned LU_DIM;
    unsigned FRAME_DIM;
	float BIN_BASE;
	float MAX_BIN;

	unordered_map<string, Parameter> params_;
	unordered_map<string, Expression> cg_params_;

public:
    explicit Pruner(ParameterCollection *model) {
    }

    explicit Pruner(SemanticOptions *semantic_options, int num_roles, SemanticDecoder *decoder, ParameterCollection *model) :
            BiLSTM(semantic_options->pruner_num_lstm_layers(),
//                   semantic_options->word_dim() + semantic_options->lemma_dim() + semantic_options->pos_dim(),
                   64,
                   semantic_options->pruner_lstm_dim(), decoder, model) {
//        WORD_DIM = semantic_options->word_dim();
//        LEMMA_DIM = semantic_options->lemma_dim();
//        POS_DIM = semantic_options->pos_dim();
	    WORD_DIM = 32;
        LEMMA_DIM = 16;
        POS_DIM = 16;
	    BIN_BASE = 2.0;
	    MAX_BIN = 4; // determined by max len and bin_base
        LSTM_DIM = semantic_options->pruner_lstm_dim();
        MLP_DIM = semantic_options->pruner_mlp_dim();

        LU_DIM = MLP_DIM;
        FRAME_DIM = MLP_DIM;
        RANK = MLP_DIM;
        DROPOUT = 0.0;
	    DROPOUT = semantic_options->dropout_rate();
	    WORD_DROPOUT = semantic_options->word_dropout_rate();
	    CHECK(DROPOUT >= 0.0 && DROPOUT < 1.0);
	    CHECK(WORD_DROPOUT >= 0.0 && WORD_DROPOUT < 1.0);
    }

	float Bin(unsigned x, bool negative) {
		CHECK_GT(x, 0);
		float ret = logf(x) / logf(BIN_BASE);
		ret = min(ret, MAX_BIN);
		if (negative) ret *= -1.0;
		return ret;
	}

    void InitParams(ParameterCollection *model);

	void StartGraph(ComputationGraph &cg, bool is_train);

    Expression BuildGraph(Instance *instance,
                          Parts *parts, vector<double> *scores,
                          const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                          unordered_map<int, int> *form_count,
                          bool is_train, ComputationGraph &cg);

    void DecodeBasicMarginals(Instance *instance, Parts *parts,
                              const vector<Expression> &scores,
                              vector<double> *predicted_output,
                              Expression &entropy, ComputationGraph &cg);
};

#endif //SRL_PRUNER_H