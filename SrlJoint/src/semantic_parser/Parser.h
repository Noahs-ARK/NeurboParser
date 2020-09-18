//
// Created by hpeng on 7/7/17.
//

#ifndef SRL_PARSER_H
#define SRL_PARSER_H

#include "H5Cpp.h"
#include "BiLSTM.h"
#include "SemanticOptions.h"
#include "SemanticPart.h"
#include "SemanticInstance.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"
#include "AlgUtils.h"

const int ELMO_DIM = 1024;
using namespace H5;
class Parser : public BiLSTM {

protected:
	unordered_map<string, Parameter> params_;
	unordered_map<string, Expression> cg_params_;
    unsigned RANK;
    unsigned LU_DIM;
    unsigned FRAME_DIM;
    unsigned ROLE_DIM;
    unsigned ROLE_SIZE;
	float BIN_BASE;
	float MAX_BIN;
	bool USE_ELMO;
	string FILE_ELMO;

public:
    explicit Parser(ParameterCollection *model) {
    }

    explicit Parser(SemanticOptions *semantic_options, int num_roles,
                    SemanticDecoder *decoder, ParameterCollection *model) :
            BiLSTM(semantic_options->num_lstm_layers(),
                   semantic_options->lstm_dim(),
                   semantic_options->lstm_dim(), decoder, model) {
        WORD_DIM = semantic_options->word_dim();
        LEMMA_DIM = semantic_options->lemma_dim();
        POS_DIM = semantic_options->pos_dim();
        LSTM_DIM = semantic_options->lstm_dim();
        MLP_DIM = semantic_options->mlp_dim();

        ROLE_SIZE = num_roles;
        ROLE_DIM = MLP_DIM;
        LU_DIM = MLP_DIM;
        FRAME_DIM = MLP_DIM;
        RANK = MLP_DIM;
	    BIN_BASE = 2.0;
	    MAX_BIN = 4; // determined by max len and bin_base
	    DROPOUT = semantic_options->dropout_rate();
	    WORD_DROPOUT = semantic_options->word_dropout_rate();
	    USE_ELMO = semantic_options->use_elmo();
	    FILE_ELMO = semantic_options->file_elmo();
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

	void ReadELMo(const string split, const int instance_id, const int slen,
	              vector<Expression> &ex_elmos, ComputationGraph &cg) {
		ex_elmos.resize(slen);
		float *data_out = new float[3 * slen * ELMO_DIM];
		const H5std_string FILE_NAME(FILE_ELMO + split);
		const H5std_string DATASET_NAME(to_string(instance_id));
		H5File file(FILE_NAME, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet(DATASET_NAME);
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_out[3];
		int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
		CHECK_EQ(dims_out[0], 3);
		CHECK_EQ(dims_out[1], slen);
		CHECK_EQ(dims_out[2], ELMO_DIM);
		dataset.read( data_out, PredType::NATIVE_FLOAT, dataspace );

		vector<float> data_buff(data_out, data_out + 3 * slen * ELMO_DIM);

//		for (int i = 0;i < 3; ++ i) {
//			for (int j = 0;j < slen; ++ j) {
//				for (int k = 0;k < ELMO_DIM; ++ k) {
//					cout << data_buff[i * slen * ELMO_DIM + j * ELMO_DIM + k] <<" ";
//				}
//				cout << endl;
//			}
//			cout << endl;
//		}
		Expression ex_elmo = transpose(reshape(input(cg, {3 * slen * ELMO_DIM}, data_buff),
		                                       {slen * ELMO_DIM, 3}));
		Expression elmo_att = cg_params_.at("elmo_att_");
		ex_elmo = elmo_att * ex_elmo; // slen * ELMO_DIM;
		ex_elmo = reshape(ex_elmo, {slen, ELMO_DIM});
		for (int i = 0;i < slen; ++ i) {
			ex_elmos[i] = pick(ex_elmo, i);
		}
		delete data_out;
	}

	void ReadWord(Instance *instance,
	              vector<Expression> &ex_word,
	              unordered_map<int, int> *form_count,
	              const string &split, int instance_id,
	              bool is_train, ComputationGraph &cg);

	void RunLSTM(Instance * instance, LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
	             const vector<Expression> &ex_words, vector<Expression> &ex_lstm,
	             bool is_train, ComputationGraph &cg);

    Expression BuildGraph(Instance *instance,
                          Parts *parts, vector<double> *scores,
                          const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                          unordered_map<int, int> *form_count, const string &split, int instance_id,
                          bool is_train, ComputationGraph &cg);
};

#endif //SRL_PARSER_H