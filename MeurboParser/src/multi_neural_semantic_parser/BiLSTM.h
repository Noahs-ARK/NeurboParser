//
// Created by hpeng on 11/3/16.
//

#ifndef CNN_BILSTM_H
#define CNN_BILSTM_H

#endif //CNN_BILSTM_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "SemanticOptions.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticDecoder.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tr1/unordered_map>
#include <time.h>

using namespace std;
using namespace dynet::expr;

const unsigned VOCAB_SIZE = 35000;
const unsigned POS_SIZE = 49;
const int UNK_ID = 0;


class biLSTM {
    //shared
public:
    dynet::LookupParameter p_embed_pre_word;
    dynet::LookupParameter p_embed_word;
    dynet::LookupParameter p_embed_pos;
    unsigned LAYERS;

    unsigned PRE_WORD_DIM;
    unsigned WORD_DIM;
    unsigned POS_DIM;
    unsigned LSTM_DIM;
    unsigned MLP_DIM;
    unsigned LABEL_SIZE;

    virtual void InitParams(dynet::Model *model) {}

    virtual Expression BuildGraph(Instance *instance, Decoder *decoder_, const bool &is_train,
                                  Parts *task1_parts, vector<double> &task1_scores,
                                  const vector<double> &task1_gold_outputs,
                                  vector<double> &task1_predicted_outputs,
                                  Parts *task2_parts, vector<double> &task2_scores,
                                  const vector<double> &task2_gold_outputs,
                                  vector<double> &task2_predicted_outputs,
                                  Parts *task3_parts, vector<double> &task3_scores,
                                  const vector<double> &task3_gold_outputs,
                                  vector<double> &task3_predicted_outputs,
                                  Parts *ctf_parts, vector<double> &ctf_scores,
                                  const vector<double> &ctf_gold_outputs,
                                  vector<double> &ctf_predicted_outputs,
                                  const bool &use_word_dropout, const float &word_dropout_rate,
                                  unordered_map<int, int> *form_count, dynet::ComputationGraph &cg) {}

    void LoadEmbedding(unordered_map<int, vector<float> > *Embedding) {
        for (auto it : (*Embedding)) {
            p_embed_pre_word.initialize(it.first, it.second);
        }
    }

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int) {

    }
};

template<class Builder>
class Shared1 : public biLSTM {
private:
    // task1
    // predicate part
    dynet::Parameter task1_p_pred_w1;
    dynet::Parameter task1_p_pred_b1;
    dynet::Parameter task1_p_pred_w2;
    dynet::Parameter task1_p_pred_b2;
    dynet::Parameter task1_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task1_p_unlab_w1_pred;
    dynet::Parameter task1_p_unlab_w1_arg;
    dynet::Parameter task1_p_unlab_b1;
    dynet::Parameter task1_p_unlab_w2;
    dynet::Parameter task1_p_unlab_b2;
    dynet::Parameter task1_p_unlab_w3;
    // labeled arc
    dynet::Parameter task1_p_lab_w1_pred;
    dynet::Parameter task1_p_lab_w1_arg;
    dynet::Parameter task1_p_lab_b1;
    dynet::Parameter task1_p_lab_w2;
    dynet::Parameter task1_p_lab_b2;
    dynet::Parameter task1_p_lab_w3;
    dynet::Parameter task1_p_lab_b3;

    // task2
    // predicate part
    dynet::Parameter task2_p_pred_w1;
    dynet::Parameter task2_p_pred_b1;
    dynet::Parameter task2_p_pred_w2;
    dynet::Parameter task2_p_pred_b2;
    dynet::Parameter task2_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task2_p_unlab_w1_pred;
    dynet::Parameter task2_p_unlab_w1_arg;
    dynet::Parameter task2_p_unlab_b1;
    dynet::Parameter task2_p_unlab_w2;
    dynet::Parameter task2_p_unlab_b2;
    dynet::Parameter task2_p_unlab_w3;
    // labeled arc
    dynet::Parameter task2_p_lab_w1_pred;
    dynet::Parameter task2_p_lab_w1_arg;
    dynet::Parameter task2_p_lab_b1;
    dynet::Parameter task2_p_lab_w2;
    dynet::Parameter task2_p_lab_b2;
    dynet::Parameter task2_p_lab_w3;
    dynet::Parameter task2_p_lab_b3;

    // task3
    // predicate part
    dynet::Parameter task3_p_pred_w1;
    dynet::Parameter task3_p_pred_b1;
    dynet::Parameter task3_p_pred_w2;
    dynet::Parameter task3_p_pred_b2;
    dynet::Parameter task3_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task3_p_unlab_w1_pred;
    dynet::Parameter task3_p_unlab_w1_arg;
    dynet::Parameter task3_p_unlab_b1;
    dynet::Parameter task3_p_unlab_w2;
    dynet::Parameter task3_p_unlab_b2;
    dynet::Parameter task3_p_unlab_w3;
    // labeled arc
    dynet::Parameter task3_p_lab_w1_pred;
    dynet::Parameter task3_p_lab_w1_arg;
    dynet::Parameter task3_p_lab_b1;
    dynet::Parameter task3_p_lab_w2;
    dynet::Parameter task3_p_lab_b2;
    dynet::Parameter task3_p_lab_w3;
    dynet::Parameter task3_p_lab_b3;

    unsigned TASK1_LABEL_SIZE;
    unsigned TASK2_LABEL_SIZE;
    unsigned TASK3_LABEL_SIZE;
    Builder l2rbuilder;
    Builder r2lbuilder;

public:
    explicit Shared1(dynet::Model *model) {
    }

    explicit Shared1(SemanticOptions *semantic_option, const int &task1_num_roles, const int &task2_num_roles,
                     const int &task3_num_roles, dynet::Model *model) :
            l2rbuilder(semantic_option->num_lstm_layers(),
                       semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                       semantic_option->lstm_dim(), *model),
            r2lbuilder(semantic_option->num_lstm_layers(),
                       semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                       semantic_option->lstm_dim(), *model) {
        LAYERS = semantic_option->num_lstm_layers();
        PRE_WORD_DIM = semantic_option->pre_word_dim();
        WORD_DIM = semantic_option->word_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->lstm_dim();
        MLP_DIM = semantic_option->mlp_dim();
        TASK1_LABEL_SIZE = task1_num_roles;
        TASK2_LABEL_SIZE = task2_num_roles;
        TASK3_LABEL_SIZE = task3_num_roles;
    }

    void InitParams(dynet::Model *model) {
        // shared
        p_embed_pre_word = model->add_lookup_parameters(VOCAB_SIZE, {PRE_WORD_DIM});
        p_embed_word = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        p_embed_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // task1
        // predicate part
        task1_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_pred_b1 = model->add_parameters({MLP_DIM});
        task1_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_pred_b2 = model->add_parameters({MLP_DIM});
        task1_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task1_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task1_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_lab_b1 = model->add_parameters({MLP_DIM});
        task1_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_lab_b2 = model->add_parameters({MLP_DIM});
        task1_p_lab_w3 = model->add_parameters({TASK1_LABEL_SIZE, MLP_DIM});
        task1_p_lab_b3 = model->add_parameters({TASK1_LABEL_SIZE});

        // task2
        // predicate part
        task2_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_pred_b1 = model->add_parameters({MLP_DIM});
        task2_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_pred_b2 = model->add_parameters({MLP_DIM});
        task2_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task2_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task2_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_lab_b1 = model->add_parameters({MLP_DIM});
        task2_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_lab_b2 = model->add_parameters({MLP_DIM});
        task2_p_lab_w3 = model->add_parameters({TASK2_LABEL_SIZE, MLP_DIM});
        task2_p_lab_b3 = model->add_parameters({TASK2_LABEL_SIZE});

        // task3
        // predicate part
        task3_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_pred_b1 = model->add_parameters({MLP_DIM});
        task3_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_pred_b2 = model->add_parameters({MLP_DIM});
        task3_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task3_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task3_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_lab_b1 = model->add_parameters({MLP_DIM});
        task3_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_lab_b2 = model->add_parameters({MLP_DIM});
        task3_p_lab_w3 = model->add_parameters({TASK3_LABEL_SIZE, MLP_DIM});
        task3_p_lab_b3 = model->add_parameters({TASK3_LABEL_SIZE});
    }


    //decoder_ -> Decode(instance, parts, scores, &predicted_outputs);
    Expression BuildGraph(Instance *instance, Decoder *decoder_, const bool &is_train,
                          Parts *task1_parts, vector<double> &task1_scores, const vector<double> &task1_gold_outputs,
                          vector<double> &task1_predicted_outputs,
                          Parts *task2_parts, vector<double> &task2_scores, const vector<double> &task2_gold_outputs,
                          vector<double> &task2_predicted_outputs,
                          Parts *task3_parts, vector<double> &task3_scores, const vector<double> &task3_gold_outputs,
                          vector<double> &task3_predicted_outputs,
                          Parts *ctf_parts, vector<double> &ctf_scores,
                          const vector<double> &ctf_gold_outputs,
                          vector<double> &ctf_predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count, dynet::ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> pos = sentence->GetPosIds();
        l2rbuilder.new_graph(cg);
        l2rbuilder.start_new_sequence();
        r2lbuilder.new_graph(cg);
        r2lbuilder.start_new_sequence();

        // task1
        // predicate part
        Expression task1_pred_w1 = parameter(cg, task1_p_pred_w1);
        Expression task1_pred_b1 = parameter(cg, task1_p_pred_b1);
        Expression task1_pred_w2 = parameter(cg, task1_p_pred_w2);
        Expression task1_pred_b2 = parameter(cg, task1_p_pred_b2);
        Expression task1_pred_w3 = parameter(cg, task1_p_pred_w3);
        // unlabeled arc
        Expression task1_unlab_w1_pred = parameter(cg, task1_p_unlab_w1_pred);
        Expression task1_unlab_w1_arg = parameter(cg, task1_p_unlab_w1_arg);
        Expression task1_unlab_b1 = parameter(cg, task1_p_unlab_b1);
        Expression task1_unlab_w2 = parameter(cg, task1_p_unlab_w2);
        Expression task1_unlab_b2 = parameter(cg, task1_p_unlab_b2);
        Expression task1_unlab_w3 = parameter(cg, task1_p_unlab_w3);
        // labeled arc
        Expression task1_lab_w1_pred = parameter(cg, task1_p_lab_w1_pred);
        Expression task1_lab_w1_arg = parameter(cg, task1_p_lab_w1_arg);
        Expression task1_lab_b1 = parameter(cg, task1_p_lab_b1);
        Expression task1_lab_w2 = parameter(cg, task1_p_lab_w2);
        Expression task1_lab_b2 = parameter(cg, task1_p_lab_b2);
        Expression task1_lab_w3 = parameter(cg, task1_p_lab_w3);
        Expression task1_lab_b3 = parameter(cg, task1_p_lab_b3);

        // task2
        // predicate part
        Expression task2_pred_w1 = parameter(cg, task2_p_pred_w1);
        Expression task2_pred_b1 = parameter(cg, task2_p_pred_b1);
        Expression task2_pred_w2 = parameter(cg, task2_p_pred_w2);
        Expression task2_pred_b2 = parameter(cg, task2_p_pred_b2);
        Expression task2_pred_w3 = parameter(cg, task2_p_pred_w3);
        // unlabeled arc
        Expression task2_unlab_w1_pred = parameter(cg, task2_p_unlab_w1_pred);
        Expression task2_unlab_w1_arg = parameter(cg, task2_p_unlab_w1_arg);
        Expression task2_unlab_b1 = parameter(cg, task2_p_unlab_b1);
        Expression task2_unlab_w2 = parameter(cg, task2_p_unlab_w2);
        Expression task2_unlab_b2 = parameter(cg, task2_p_unlab_b2);
        Expression task2_unlab_w3 = parameter(cg, task2_p_unlab_w3);
        // labeled arc
        Expression task2_lab_w1_pred = parameter(cg, task2_p_lab_w1_pred);
        Expression task2_lab_w1_arg = parameter(cg, task2_p_lab_w1_arg);
        Expression task2_lab_b1 = parameter(cg, task2_p_lab_b1);
        Expression task2_lab_w2 = parameter(cg, task2_p_lab_w2);
        Expression task2_lab_b2 = parameter(cg, task2_p_lab_b2);
        Expression task2_lab_w3 = parameter(cg, task2_p_lab_w3);
        Expression task2_lab_b3 = parameter(cg, task2_p_lab_b3);

        // task3
        // predicate part
        Expression task3_pred_w1 = parameter(cg, task3_p_pred_w1);
        Expression task3_pred_b1 = parameter(cg, task3_p_pred_b1);
        Expression task3_pred_w2 = parameter(cg, task3_p_pred_w2);
        Expression task3_pred_b2 = parameter(cg, task3_p_pred_b2);
        Expression task3_pred_w3 = parameter(cg, task3_p_pred_w3);
        // unlabeled arc
        Expression task3_unlab_w1_pred = parameter(cg, task3_p_unlab_w1_pred);
        Expression task3_unlab_w1_arg = parameter(cg, task3_p_unlab_w1_arg);
        Expression task3_unlab_b1 = parameter(cg, task3_p_unlab_b1);
        Expression task3_unlab_w2 = parameter(cg, task3_p_unlab_w2);
        Expression task3_unlab_b2 = parameter(cg, task3_p_unlab_b2);
        Expression task3_unlab_w3 = parameter(cg, task3_p_unlab_w3);
        // labeled arc
        Expression task3_lab_w1_pred = parameter(cg, task3_p_lab_w1_pred);
        Expression task3_lab_w1_arg = parameter(cg, task3_p_lab_w1_arg);
        Expression task3_lab_b1 = parameter(cg, task3_p_lab_b1);
        Expression task3_lab_w2 = parameter(cg, task3_p_lab_w2);
        Expression task3_lab_b2 = parameter(cg, task3_p_lab_b2);
        Expression task3_lab_w3 = parameter(cg, task3_p_lab_w3);
        Expression task3_lab_b3 = parameter(cg, task3_p_lab_b3);

        vector<Expression> ex_words(slen), i_errs;
        vector<Expression> ex_l2r(slen), ex_r2l(slen);
        for (int i = 0; i < slen; ++i) {
            int word_idx = words[i];
            if (use_word_dropout && word_idx != UNK_ID) {
                int count = form_count->find(word_idx)->second;
                float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                if (rand_float < word_dropout_rate / (static_cast<float> (count) + word_dropout_rate))
                    word_idx = UNK_ID;
            }
            Expression x_pre_word = lookup(cg, p_embed_pre_word, word_idx);
            Expression x_word = lookup(cg, p_embed_word, word_idx);
            Expression x_pos = lookup(cg, p_embed_pos, pos[i]);
            ex_words[i] = concatenate({x_pre_word, x_word, x_pos});
            ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
        }
        for (int i = 0; i < slen; ++i) {
            ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
        }

        vector<Expression> task1_unlab_pred_exs, task1_unlab_arg_exs;
        vector<Expression> task1_lab_pred_exs, task1_lab_arg_exs;
        vector<Expression> task2_unlab_pred_exs, task2_unlab_arg_exs;
        vector<Expression> task2_lab_pred_exs, task2_lab_arg_exs;
        vector<Expression> task3_unlab_pred_exs, task3_unlab_arg_exs;
        vector<Expression> task3_lab_pred_exs, task3_lab_arg_exs;
        for (int i = 0; i < slen; ++i) {
            Expression word_ex = concatenate({ex_l2r[i], ex_r2l[i]});
            task1_unlab_pred_exs.push_back(task1_unlab_w1_pred * word_ex);
            task1_unlab_arg_exs.push_back(task1_unlab_w1_arg * word_ex);
            task1_lab_pred_exs.push_back(task1_lab_w1_pred * word_ex);
            task1_lab_arg_exs.push_back(task1_lab_w1_arg * word_ex);

            task2_unlab_pred_exs.push_back(task2_unlab_w1_pred * word_ex);
            task2_unlab_arg_exs.push_back(task2_unlab_w1_arg * word_ex);
            task2_lab_pred_exs.push_back(task2_lab_w1_pred * word_ex);
            task2_lab_arg_exs.push_back(task2_lab_w1_arg * word_ex);

            task3_unlab_pred_exs.push_back(task3_unlab_w1_pred * word_ex);
            task3_unlab_arg_exs.push_back(task3_unlab_w1_arg * word_ex);
            task3_lab_pred_exs.push_back(task3_lab_w1_pred * word_ex);
            task3_lab_arg_exs.push_back(task3_lab_w1_arg * word_ex);
        }

        /* task1 begin */
        vector<Expression> task1_exps(task1_parts->size());
        task1_scores.assign(task1_parts->size(), 0.0);
        task1_predicted_outputs.assign(task1_parts->size(), 0.0);
        SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
        for (int r = 0; r < task1_parts->size(); ++r) {
            if ((*task1_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task1_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task1_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task1_pred_MLP_in = tanh(task1_pred_w1 * task1_pred_ex + task1_pred_b1);
                Expression task1_pred_phi = tanh(task1_pred_w2 * task1_pred_MLP_in + task1_pred_b2);
                task1_exps[r] = task1_pred_w3 * task1_pred_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
            } else if ((*task1_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task1_unlab_MLP_in = tanh(
                        task1_unlab_pred_exs[idx_pred] + task1_unlab_arg_exs[idx_arg] + task1_unlab_b1);
                Expression task1_lab_MLP_in = tanh(
                        task1_lab_pred_exs[idx_pred] + task1_lab_arg_exs[idx_arg] + task1_lab_b1);
                Expression task1_unlab_phi = tanh(task1_unlab_w2 * task1_unlab_MLP_in + task1_unlab_b2);
                Expression task1_lab_phi = tanh(task1_lab_w2 * task1_lab_MLP_in + task1_lab_b2);
                task1_exps[r] = task1_unlab_w3 * task1_unlab_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
                Expression lab_MLP_out = task1_lab_w3 * task1_lab_phi + task1_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task1_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task1_parts->size());
                    CHECK_EQ((*task1_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task1_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task1_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task1_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task1 end */

        /* task2 begin */
        vector<Expression> task2_exps(task2_parts->size());
        task2_scores.assign(task2_parts->size(), 0.0);
        task2_predicted_outputs.assign(task2_parts->size(), 0.0);
        SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
        for (int r = 0; r < task2_parts->size(); ++r) {
            if ((*task2_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task2_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task2_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task2_pred_MLP_in = tanh(task2_pred_w1 * task2_pred_ex + task2_pred_b1);
                Expression task2_pred_phi = tanh(task2_pred_w2 * task2_pred_MLP_in + task2_pred_b2);
                task2_exps[r] = task2_pred_w3 * task2_pred_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
            } else if ((*task2_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task2_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task2_unlab_MLP_in = tanh(
                        task2_unlab_pred_exs[idx_pred] + task2_unlab_arg_exs[idx_arg] + task2_unlab_b1);
                Expression task2_lab_MLP_in = tanh(
                        task2_lab_pred_exs[idx_pred] + task2_lab_arg_exs[idx_arg] + task2_lab_b1);
                Expression task2_unlab_phi = tanh(task2_unlab_w2 * task2_unlab_MLP_in + task2_unlab_b2);
                Expression task2_lab_phi = tanh(task2_lab_w2 * task2_lab_MLP_in + task2_lab_b2);
                task2_exps[r] = task2_unlab_w3 * task2_unlab_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
                Expression lab_MLP_out = task2_lab_w3 * task2_lab_phi + task2_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task2_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task2_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task2_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task2 end */

        /* task3 begin */
        vector<Expression> task3_exps(task3_parts->size());
        task3_scores.assign(task3_parts->size(), 0.0);
        task3_predicted_outputs.assign(task3_parts->size(), 0.0);
        SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
        for (int r = 0; r < task3_parts->size(); ++r) {
            if ((*task3_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task3_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task3_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task3_pred_MLP_in = tanh(task3_pred_w1 * task3_pred_ex + task3_pred_b1);
                Expression task3_pred_phi = tanh(task3_pred_w2 * task3_pred_MLP_in + task3_pred_b2);
                task3_exps[r] = task3_pred_w3 * task3_pred_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
            } else if ((*task3_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task3_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task3_unlab_MLP_in = tanh(
                        task3_unlab_pred_exs[idx_pred] + task3_unlab_arg_exs[idx_arg] + task3_unlab_b1);
                Expression task3_lab_MLP_in = tanh(
                        task3_lab_pred_exs[idx_pred] + task3_lab_arg_exs[idx_arg] + task3_lab_b1);
                Expression task3_unlab_phi = tanh(task3_unlab_w2 * task3_unlab_MLP_in + task3_unlab_b2);
                Expression task3_lab_phi = tanh(task3_lab_w2 * task3_lab_MLP_in + task3_lab_b2);
                task3_exps[r] = task3_unlab_w3 * task3_unlab_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
                Expression lab_MLP_out = task3_lab_w3 * task3_lab_phi + task3_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task3_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task3_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task3_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task3 end */

        if (!is_train) {
            decoder_->Decode(instance, task1_parts, task1_scores, &task1_predicted_outputs);
            decoder_->Decode(instance, task2_parts, task2_scores, &task2_predicted_outputs);
            decoder_->Decode(instance, task3_parts, task3_scores, &task3_predicted_outputs);
            for (int r = 0; r < task1_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            for (int r = 0; r < task2_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < task3_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            Expression loss = Expression(&cg, cg.add_input(0.0));
            if (i_errs.size() > 0) {
                loss = loss + sum(i_errs);
            }
            return loss;
        }

        double s_loss = 0.0, s_cost = 0.0, cost = 0.0;
        decoder_->DecodeCostAugmented(instance, task1_parts, task1_scores, task1_gold_outputs,
                                      &task1_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;
        decoder_->DecodeCostAugmented(instance, task2_parts, task2_scores, task2_gold_outputs,
                                      &task2_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;
        decoder_->DecodeCostAugmented(instance, task3_parts, task3_scores, task3_gold_outputs,
                                      &task3_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;

        for (int r = 0; r < task1_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task2_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task3_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                i_errs.push_back(i_err);
            }
        }
        Expression loss = Expression(&cg, cg.add_input(cost));
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};

template<class Builder>
class Shared3 : public biLSTM {
private:
    // task1
    // predicate part
    dynet::Parameter task1_p_pred_w1;
    dynet::Parameter task1_p_pred_b1;
    dynet::Parameter task1_p_pred_w2;
    dynet::Parameter task1_p_pred_b2;
    dynet::Parameter task1_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task1_p_unlab_w1_pred;
    dynet::Parameter task1_p_unlab_w1_arg;
    dynet::Parameter task1_p_unlab_b1;
    dynet::Parameter task1_p_unlab_w2;
    dynet::Parameter task1_p_unlab_b2;
    dynet::Parameter task1_p_unlab_w3;
    // labeled arc
    dynet::Parameter task1_p_lab_w1_pred;
    dynet::Parameter task1_p_lab_w1_arg;
    dynet::Parameter task1_p_lab_b1;
    dynet::Parameter task1_p_lab_w2;
    dynet::Parameter task1_p_lab_b2;
    dynet::Parameter task1_p_lab_w3;
    dynet::Parameter task1_p_lab_b3;

    // task2
    // predicate part
    dynet::Parameter task2_p_pred_w1;
    dynet::Parameter task2_p_pred_b1;
    dynet::Parameter task2_p_pred_w2;
    dynet::Parameter task2_p_pred_b2;
    dynet::Parameter task2_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task2_p_unlab_w1_pred;
    dynet::Parameter task2_p_unlab_w1_arg;
    dynet::Parameter task2_p_unlab_b1;
    dynet::Parameter task2_p_unlab_w2;
    dynet::Parameter task2_p_unlab_b2;
    dynet::Parameter task2_p_unlab_w3;
    // labeled arc
    dynet::Parameter task2_p_lab_w1_pred;
    dynet::Parameter task2_p_lab_w1_arg;
    dynet::Parameter task2_p_lab_b1;
    dynet::Parameter task2_p_lab_w2;
    dynet::Parameter task2_p_lab_b2;
    dynet::Parameter task2_p_lab_w3;
    dynet::Parameter task2_p_lab_b3;

    // task3
    // predicate part
    dynet::Parameter task3_p_pred_w1;
    dynet::Parameter task3_p_pred_b1;
    dynet::Parameter task3_p_pred_w2;
    dynet::Parameter task3_p_pred_b2;
    dynet::Parameter task3_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task3_p_unlab_w1_pred;
    dynet::Parameter task3_p_unlab_w1_arg;
    dynet::Parameter task3_p_unlab_b1;
    dynet::Parameter task3_p_unlab_w2;
    dynet::Parameter task3_p_unlab_b2;
    dynet::Parameter task3_p_unlab_w3;
    // labeled arc
    dynet::Parameter task3_p_lab_w1_pred;
    dynet::Parameter task3_p_lab_w1_arg;
    dynet::Parameter task3_p_lab_b1;
    dynet::Parameter task3_p_lab_w2;
    dynet::Parameter task3_p_lab_b2;
    dynet::Parameter task3_p_lab_w3;
    dynet::Parameter task3_p_lab_b3;

    // ctf
    // unlabeled
    dynet::Parameter task1_p_unlab_U;
    dynet::Parameter task2_p_unlab_U;
    dynet::Parameter task3_p_unlab_U;
    dynet::Parameter task1_p_unlab_V;
    dynet::Parameter task2_p_unlab_V;
    dynet::Parameter task3_p_unlab_V;
    // labeled
    dynet::Parameter task1_p_lab_U;
    dynet::Parameter task2_p_lab_U;
    dynet::Parameter task3_p_lab_U;
    dynet::Parameter task1_p_lab_V;
    dynet::Parameter task2_p_lab_V;
    dynet::Parameter task3_p_lab_V;

    unsigned TASK1_LABEL_SIZE;
    unsigned TASK2_LABEL_SIZE;
    unsigned TASK3_LABEL_SIZE;
    unsigned RANK;
    Builder l2rbuilder;
    Builder r2lbuilder;

public:
    explicit Shared3(dynet::Model *model) {

    }

    explicit Shared3(SemanticOptions *semantic_option, const int &task1_num_roles, const int &task2_num_roles,
                     const int &task3_num_roles, dynet::Model *model) :
            l2rbuilder(semantic_option->num_lstm_layers(),
                       semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                       semantic_option->lstm_dim(), *model),
            r2lbuilder(semantic_option->num_lstm_layers(),
                       semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                       semantic_option->lstm_dim(), *model) {
        LAYERS = semantic_option->num_lstm_layers();
        PRE_WORD_DIM = semantic_option->pre_word_dim();
        WORD_DIM = semantic_option->word_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->lstm_dim();
        MLP_DIM = semantic_option->mlp_dim();
        TASK1_LABEL_SIZE = task1_num_roles;
        TASK2_LABEL_SIZE = task2_num_roles;
        TASK3_LABEL_SIZE = task3_num_roles;
        RANK = semantic_option->rank();
    }

    void InitParams(dynet::Model *model) {
        // shared
        p_embed_pre_word = model->add_lookup_parameters(VOCAB_SIZE, {PRE_WORD_DIM});
        p_embed_word = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        p_embed_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // task1
        // predicate part
        task1_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_pred_b1 = model->add_parameters({MLP_DIM});
        task1_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_pred_b2 = model->add_parameters({MLP_DIM});
        task1_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task1_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task1_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task1_p_lab_b1 = model->add_parameters({MLP_DIM});
        task1_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_lab_b2 = model->add_parameters({MLP_DIM});
        task1_p_lab_w3 = model->add_parameters({TASK1_LABEL_SIZE, MLP_DIM});
        task1_p_lab_b3 = model->add_parameters({TASK1_LABEL_SIZE});

        // task2
        // predicate part
        task2_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_pred_b1 = model->add_parameters({MLP_DIM});
        task2_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_pred_b2 = model->add_parameters({MLP_DIM});
        task2_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task2_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task2_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task2_p_lab_b1 = model->add_parameters({MLP_DIM});
        task2_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_lab_b2 = model->add_parameters({MLP_DIM});
        task2_p_lab_w3 = model->add_parameters({TASK2_LABEL_SIZE, MLP_DIM});
        task2_p_lab_b3 = model->add_parameters({TASK2_LABEL_SIZE});

        // task3
        // predicate part
        task3_p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_pred_b1 = model->add_parameters({MLP_DIM});
        task3_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_pred_b2 = model->add_parameters({MLP_DIM});
        task3_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task3_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task3_p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        task3_p_lab_b1 = model->add_parameters({MLP_DIM});
        task3_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_lab_b2 = model->add_parameters({MLP_DIM});
        task3_p_lab_w3 = model->add_parameters({TASK3_LABEL_SIZE, MLP_DIM});
        task3_p_lab_b3 = model->add_parameters({TASK3_LABEL_SIZE});

        // ctf
        // unlabeled
        task1_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task2_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task3_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task1_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        task2_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        task3_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        // labeled
        task1_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task2_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task3_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task1_p_lab_V = model->add_parameters({RANK, MLP_DIM});
        task2_p_lab_V = model->add_parameters({RANK, MLP_DIM});
        task3_p_lab_V = model->add_parameters({RANK, MLP_DIM});
    }


    //decoder_ -> Decode(instance, parts, scores, &predicted_outputs);
    Expression BuildGraph(Instance *instance, Decoder *decoder_, const bool &is_train,
                          Parts *task1_parts, vector<double> &task1_scores, const vector<double> &task1_gold_outputs,
                          vector<double> &task1_predicted_outputs,
                          Parts *task2_parts, vector<double> &task2_scores, const vector<double> &task2_gold_outputs,
                          vector<double> &task2_predicted_outputs,
                          Parts *task3_parts, vector<double> &task3_scores, const vector<double> &task3_gold_outputs,
                          vector<double> &task3_predicted_outputs,
                          Parts *ctf_parts, vector<double> &ctf_scores, const vector<double> &ctf_gold_outputs,
                          vector<double> &ctf_predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count, dynet::ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> pos = sentence->GetPosIds();
        l2rbuilder.new_graph(cg);
        l2rbuilder.start_new_sequence();
        r2lbuilder.new_graph(cg);
        r2lbuilder.start_new_sequence();

        // task1
        // predicate part
        Expression task1_pred_w1 = parameter(cg, task1_p_pred_w1);
        Expression task1_pred_b1 = parameter(cg, task1_p_pred_b1);
        Expression task1_pred_w2 = parameter(cg, task1_p_pred_w2);
        Expression task1_pred_b2 = parameter(cg, task1_p_pred_b2);
        Expression task1_pred_w3 = parameter(cg, task1_p_pred_w3);
        // unlabeled arc
        Expression task1_unlab_w1_pred = parameter(cg, task1_p_unlab_w1_pred);
        Expression task1_unlab_w1_arg = parameter(cg, task1_p_unlab_w1_arg);
        Expression task1_unlab_b1 = parameter(cg, task1_p_unlab_b1);
        Expression task1_unlab_w2 = parameter(cg, task1_p_unlab_w2);
        Expression task1_unlab_b2 = parameter(cg, task1_p_unlab_b2);
        Expression task1_unlab_w3 = parameter(cg, task1_p_unlab_w3);
        // labeled arc
        Expression task1_lab_w1_pred = parameter(cg, task1_p_lab_w1_pred);
        Expression task1_lab_w1_arg = parameter(cg, task1_p_lab_w1_arg);
        Expression task1_lab_b1 = parameter(cg, task1_p_lab_b1);
        Expression task1_lab_w2 = parameter(cg, task1_p_lab_w2);
        Expression task1_lab_b2 = parameter(cg, task1_p_lab_b2);
        Expression task1_lab_w3 = parameter(cg, task1_p_lab_w3);
        Expression task1_lab_b3 = parameter(cg, task1_p_lab_b3);

        // task2
        // predicate part
        Expression task2_pred_w1 = parameter(cg, task2_p_pred_w1);
        Expression task2_pred_b1 = parameter(cg, task2_p_pred_b1);
        Expression task2_pred_w2 = parameter(cg, task2_p_pred_w2);
        Expression task2_pred_b2 = parameter(cg, task2_p_pred_b2);
        Expression task2_pred_w3 = parameter(cg, task2_p_pred_w3);
        // unlabeled arc
        Expression task2_unlab_w1_pred = parameter(cg, task2_p_unlab_w1_pred);
        Expression task2_unlab_w1_arg = parameter(cg, task2_p_unlab_w1_arg);
        Expression task2_unlab_b1 = parameter(cg, task2_p_unlab_b1);
        Expression task2_unlab_w2 = parameter(cg, task2_p_unlab_w2);
        Expression task2_unlab_b2 = parameter(cg, task2_p_unlab_b2);
        Expression task2_unlab_w3 = parameter(cg, task2_p_unlab_w3);
        // labeled arc
        Expression task2_lab_w1_pred = parameter(cg, task2_p_lab_w1_pred);
        Expression task2_lab_w1_arg = parameter(cg, task2_p_lab_w1_arg);
        Expression task2_lab_b1 = parameter(cg, task2_p_lab_b1);
        Expression task2_lab_w2 = parameter(cg, task2_p_lab_w2);
        Expression task2_lab_b2 = parameter(cg, task2_p_lab_b2);
        Expression task2_lab_w3 = parameter(cg, task2_p_lab_w3);
        Expression task2_lab_b3 = parameter(cg, task2_p_lab_b3);

        // task3
        // predicate part
        Expression task3_pred_w1 = parameter(cg, task3_p_pred_w1);
        Expression task3_pred_b1 = parameter(cg, task3_p_pred_b1);
        Expression task3_pred_w2 = parameter(cg, task3_p_pred_w2);
        Expression task3_pred_b2 = parameter(cg, task3_p_pred_b2);
        Expression task3_pred_w3 = parameter(cg, task3_p_pred_w3);
        // unlabeled arc
        Expression task3_unlab_w1_pred = parameter(cg, task3_p_unlab_w1_pred);
        Expression task3_unlab_w1_arg = parameter(cg, task3_p_unlab_w1_arg);
        Expression task3_unlab_b1 = parameter(cg, task3_p_unlab_b1);
        Expression task3_unlab_w2 = parameter(cg, task3_p_unlab_w2);
        Expression task3_unlab_b2 = parameter(cg, task3_p_unlab_b2);
        Expression task3_unlab_w3 = parameter(cg, task3_p_unlab_w3);
        // labeled arc
        Expression task3_lab_w1_pred = parameter(cg, task3_p_lab_w1_pred);
        Expression task3_lab_w1_arg = parameter(cg, task3_p_lab_w1_arg);
        Expression task3_lab_b1 = parameter(cg, task3_p_lab_b1);
        Expression task3_lab_w2 = parameter(cg, task3_p_lab_w2);
        Expression task3_lab_b2 = parameter(cg, task3_p_lab_b2);
        Expression task3_lab_w3 = parameter(cg, task3_p_lab_w3);
        Expression task3_lab_b3 = parameter(cg, task3_p_lab_b3);

        // ctf
        Expression task1_unlab_U = parameter(cg, task1_p_unlab_U);
        Expression task2_unlab_U = parameter(cg, task2_p_unlab_U);
        Expression task3_unlab_U = parameter(cg, task3_p_unlab_U);
        Expression task1_unlab_V = parameter(cg, task1_p_unlab_V);
        Expression task2_unlab_V = parameter(cg, task2_p_unlab_V);
        Expression task3_unlab_V = parameter(cg, task3_p_unlab_V);
        Expression task1_lab_U = parameter(cg, task1_p_lab_U);
        Expression task2_lab_U = parameter(cg, task2_p_lab_U);
        Expression task3_lab_U = parameter(cg, task3_p_lab_U);
        Expression task1_lab_V = parameter(cg, task1_p_lab_V);
        Expression task2_lab_V = parameter(cg, task2_p_lab_V);
        Expression task3_lab_V = parameter(cg, task3_p_lab_V);

        vector<Expression> ex_words(slen);
        vector<Expression> ex_l2r(slen);
        vector<Expression> ex_r2l(slen);
        vector<Expression> i_errs;
        for (int i = 0; i < slen; ++i) {
            int word_idx = words[i];
            if (use_word_dropout && word_idx != UNK_ID) {
                int count = form_count->find(word_idx)->second;
                float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                if (rand_float < word_dropout_rate / (static_cast<float> (count) + word_dropout_rate))
                    word_idx = UNK_ID;
            }
            Expression x_pre_word = lookup(cg, p_embed_pre_word, word_idx);
            Expression x_word = lookup(cg, p_embed_word, word_idx);
            Expression x_pos = lookup(cg, p_embed_pos, pos[i]);
            ex_words[i] = concatenate({x_pre_word, x_word, x_pos});
            ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
        }
        for (int i = 0; i < slen; ++i) {
            ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
        }

        vector<Expression> task1_unlab_pred_exs, task1_unlab_arg_exs;
        vector<Expression> task1_lab_pred_exs, task1_lab_arg_exs;
        vector<Expression> task2_unlab_pred_exs, task2_unlab_arg_exs;
        vector<Expression> task2_lab_pred_exs, task2_lab_arg_exs;
        vector<Expression> task3_unlab_pred_exs, task3_unlab_arg_exs;
        vector<Expression> task3_lab_pred_exs, task3_lab_arg_exs;
        for (int i = 0; i < slen; ++i) {
            Expression word_ex = concatenate({ex_l2r[i], ex_r2l[i]});
            task1_unlab_pred_exs.push_back(task1_unlab_w1_pred * word_ex);
            task1_unlab_arg_exs.push_back(task1_unlab_w1_arg * word_ex);
            task1_lab_pred_exs.push_back(task1_lab_w1_pred * word_ex);
            task1_lab_arg_exs.push_back(task1_lab_w1_arg * word_ex);

            task2_unlab_pred_exs.push_back(task2_unlab_w1_pred * word_ex);
            task2_unlab_arg_exs.push_back(task2_unlab_w1_arg * word_ex);
            task2_lab_pred_exs.push_back(task2_lab_w1_pred * word_ex);
            task2_lab_arg_exs.push_back(task2_lab_w1_arg * word_ex);

            task3_unlab_pred_exs.push_back(task3_unlab_w1_pred * word_ex);
            task3_unlab_arg_exs.push_back(task3_unlab_w1_arg * word_ex);
            task3_lab_pred_exs.push_back(task3_lab_w1_pred * word_ex);
            task3_lab_arg_exs.push_back(task3_lab_w1_arg * word_ex);
        }

        /* task1 begin */
        vector<Expression> task1_exps(task1_parts->size());
        task1_scores.assign(task1_parts->size(), 0.0);
        task1_predicted_outputs.assign(task1_parts->size(), 0.0);
        SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
        vector<Expression> task1_unlab_phi_cache(slen * slen);
        vector<bool> task1_unlab_phi_flag(slen * slen, false);
        vector<Expression> task1_lab_phi_cache(slen * slen);
        vector<bool> task1_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task1_parts->size(); ++r) {
            if ((*task1_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task1_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task1_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task1_pred_MLP_in = tanh(task1_pred_w1 * task1_pred_ex + task1_pred_b1);
                Expression task1_pred_phi = tanh(task1_pred_w2 * task1_pred_MLP_in + task1_pred_b2);
                task1_exps[r] = task1_pred_w3 * task1_pred_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
            } else if ((*task1_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task1_unlab_MLP_in = tanh(
                        task1_unlab_pred_exs[idx_pred] + task1_unlab_arg_exs[idx_arg] + task1_unlab_b1);
                Expression task1_lab_MLP_in = tanh(
                        task1_lab_pred_exs[idx_pred] + task1_lab_arg_exs[idx_arg] + task1_lab_b1);
                Expression task1_unlab_phi = tanh(task1_unlab_w2 * task1_unlab_MLP_in + task1_unlab_b2);
                Expression task1_lab_phi = tanh(task1_lab_w2 * task1_lab_MLP_in + task1_lab_b2);
                task1_unlab_phi_cache[idx_pred * slen + idx_arg] = task1_unlab_phi;
                task1_lab_phi_cache[idx_pred * slen + idx_arg] = task1_lab_phi;
                task1_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task1_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task1_exps[r] = task1_unlab_w3 * task1_unlab_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
                Expression lab_MLP_out = task1_lab_w3 * task1_lab_phi + task1_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task1_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task1_parts->size());
                    CHECK_EQ((*task1_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task1_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task1_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task1_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task1 end */

        /* task2 begin */
        vector<Expression> task2_exps(task2_parts->size());
        task2_scores.assign(task2_parts->size(), 0.0);
        task2_predicted_outputs.assign(task2_parts->size(), 0.0);
        SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
        vector<Expression> task2_unlab_phi_cache(slen * slen);
        vector<bool> task2_unlab_phi_flag(slen * slen, false);
        vector<Expression> task2_lab_phi_cache(slen * slen);
        vector<bool> task2_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task2_parts->size(); ++r) {
            if ((*task2_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task2_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task2_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task2_pred_MLP_in = tanh(task2_pred_w1 * task2_pred_ex + task2_pred_b1);
                Expression task2_pred_phi = tanh(task2_pred_w2 * task2_pred_MLP_in + task2_pred_b2);
                task2_exps[r] = task2_pred_w3 * task2_pred_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
            } else if ((*task2_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task2_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task2_unlab_MLP_in = tanh(
                        task2_unlab_pred_exs[idx_pred] + task2_unlab_arg_exs[idx_arg] + task2_unlab_b1);
                Expression task2_lab_MLP_in = tanh(
                        task2_lab_pred_exs[idx_pred] + task2_lab_arg_exs[idx_arg] + task2_lab_b1);
                Expression task2_unlab_phi = tanh(task2_unlab_w2 * task2_unlab_MLP_in + task2_unlab_b2);
                Expression task2_lab_phi = tanh(task2_lab_w2 * task2_lab_MLP_in + task2_lab_b2);
                task2_unlab_phi_cache[idx_pred * slen + idx_arg] = task2_unlab_phi;
                task2_lab_phi_cache[idx_pred * slen + idx_arg] = task2_lab_phi;
                task2_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task2_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task2_exps[r] = task2_unlab_w3 * task2_unlab_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
                Expression lab_MLP_out = task2_lab_w3 * task2_lab_phi + task2_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task2_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task2_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task2_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task2 end */

        /* task3 begin */
        vector<Expression> task3_exps(task3_parts->size());
        task3_scores.assign(task3_parts->size(), 0.0);
        task3_predicted_outputs.assign(task3_parts->size(), 0.0);
        SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
        vector<Expression> task3_unlab_phi_cache(slen * slen);
        vector<bool> task3_unlab_phi_flag(slen * slen, false);
        vector<Expression> task3_lab_phi_cache(slen * slen);
        vector<bool> task3_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task3_parts->size(); ++r) {
            if ((*task3_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task3_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task3_pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression task3_pred_MLP_in = tanh(task3_pred_w1 * task3_pred_ex + task3_pred_b1);
                Expression task3_pred_phi = tanh(task3_pred_w2 * task3_pred_MLP_in + task3_pred_b2);
                task3_exps[r] = task3_pred_w3 * task3_pred_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
            } else if ((*task3_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task3_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task3_unlab_MLP_in = tanh(
                        task3_unlab_pred_exs[idx_pred] + task3_unlab_arg_exs[idx_arg] + task3_unlab_b1);
                Expression task3_lab_MLP_in = tanh(
                        task3_lab_pred_exs[idx_pred] + task3_lab_arg_exs[idx_arg] + task3_lab_b1);
                Expression task3_unlab_phi = tanh(task3_unlab_w2 * task3_unlab_MLP_in + task3_unlab_b2);
                Expression task3_lab_phi = tanh(task3_lab_w2 * task3_lab_MLP_in + task3_lab_b2);
                task3_unlab_phi_cache[idx_pred * slen + idx_arg] = task3_unlab_phi;
                task3_lab_phi_cache[idx_pred * slen + idx_arg] = task3_lab_phi;
                task3_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task3_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task3_exps[r] = task3_unlab_w3 * task3_unlab_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
                Expression lab_MLP_out = task3_lab_w3 * task3_lab_phi + task3_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task3_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task3_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task3_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task3 end */

        /* ctf begin */
        vector<Expression> ctf_exps(ctf_parts->size());
        ctf_scores.assign(ctf_parts->size(), 0.0);
        ctf_predicted_outputs.assign(ctf_parts->size(), 0.0);
        Expression task1task2_unlab_psi_cache = cmult(task1_unlab_V * transpose(task1_unlab_w3),
                                                      task2_unlab_V * transpose(task2_unlab_w3));
        Expression task1task3_unlab_psi_cache = cmult(task1_unlab_V * transpose(task1_unlab_w3),
                                                      task3_unlab_V * transpose(task3_unlab_w3));
        Expression task2task3_unlab_psi_cache = cmult(task2_unlab_V * transpose(task2_unlab_w3),
                                                      task3_unlab_V * transpose(task3_unlab_w3));
        Expression all_unlab_psi_cache = cmult(task1task2_unlab_psi_cache,
                                               task3_unlab_V * transpose(task3_unlab_w3));

        vector<Expression> task1task2_lab_phi_cache(slen * slen);
        vector<bool> task1task2_lab_phi_flag(slen * slen, false);
        vector<Expression> task1task3_lab_phi_cache(slen * slen);
        vector<bool> task1task3_lab_phi_flag(slen * slen, false);
        vector<Expression> task2task3_lab_phi_cache(slen * slen);
        vector<bool> task2task3_lab_phi_flag(slen * slen, false);
        vector<Expression> all_lab_phi_cache(slen * slen);
        vector<bool> all_lab_phi_flag(slen * slen, false);

        Expression task1_lab_psi_cache = task1_lab_w3 * transpose(task1_lab_V);
        Expression task2_lab_psi_cache = task2_lab_w3 * transpose(task2_lab_V);
        Expression task3_lab_psi_cache = task3_lab_w3 * transpose(task3_lab_V);

        vector<Expression> task1task2_lab_psi_cache(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE);
        vector<bool> task1task2_lab_psi_flag(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE, false);
        vector<Expression> task1task3_lab_psi_cache(TASK1_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> task1task3_lab_psi_flag(TASK1_LABEL_SIZE * TASK3_LABEL_SIZE, false);
        vector<Expression> task2task3_lab_psi_cache(TASK2_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> task2task3_lab_psi_flag(TASK2_LABEL_SIZE * TASK3_LABEL_SIZE, false);
        vector<Expression> all_lab_psi_cache(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> all_lab_psi_flag(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE, false);

        for (int r = 0; r < ctf_parts->size(); ++r) {
            if ((*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORM2NDORDER
                || (*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORM3RDORDER) {
                SemanticPartCrossForm2ndOrder *cross_form_arc = static_cast<SemanticPartCrossForm2ndOrder *>((*ctf_parts)[r]);
                int idx_pred = cross_form_arc->predicate();
                int idx_arg = cross_form_arc->argument();
                int idx = idx_pred * slen + idx_arg;
                int task1_role = cross_form_arc->role("task1");
                int task2_role = cross_form_arc->role("task2");
                int task3_role = cross_form_arc->role("task3");
                CHECK((task1_role < 0 || task1_unlab_phi_flag[idx])
                      && (task2_role < 0 || task2_unlab_phi_flag[idx])
                      && (task3_role < 0 || task3_unlab_phi_flag[idx]));

                if (task1_role > 0 && task2_role > 0 && task3_role < 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task2_unlab_U * task2_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task1task2_unlab_psi_cache)));
                } else if (task1_role > 0 && task2_role < 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task1task3_unlab_psi_cache)));
                } else if (task1_role < 0 && task2_role > 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task2_unlab_U * task2_unlab_phi_cache[idx],
                                               task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task2task3_unlab_psi_cache)));
                } else if (task1_role > 0 && task2_role > 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task2_unlab_U * task2_unlab_phi_cache[idx]);
                    t_unlab = cmult(t_unlab, task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, all_unlab_psi_cache)));
                } else {
                    CHECK(false) << "Cross form arc error. Giving up.";
                }
                ctf_scores[r] = dynet::as_scalar(cg.incremental_forward(ctf_exps[r]));
            } else if ((*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORMLABELED2NDORDER
                       || (*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORMLABELED3RDORDER) {
                SemanticPartCrossFormLabeled2ndOrder *cross_form_labeled_arc
                        = static_cast<SemanticPartCrossFormLabeled2ndOrder *>((*ctf_parts)[r]);
                int idx_pred = cross_form_labeled_arc->predicate();
                int idx_arg = cross_form_labeled_arc->argument();
                int idx = idx_pred * slen + idx_arg;
                int task1_role = cross_form_labeled_arc->role("task1");
                int task2_role = cross_form_labeled_arc->role("task2");
                int task3_role = cross_form_labeled_arc->role("task3");
                CHECK((task1_role < 0 || task1_lab_phi_flag[idx])
                      && (task2_role < 0 || task2_lab_phi_flag[idx])
                      && (task3_role < 0 || task3_lab_phi_flag[idx]));

                if (task1_role >= 0 && task2_role >= 0 && task3_role < 0) {
                    if (!task1task2_lab_phi_flag[idx]) {
                        task1task2_lab_phi_cache[idx] = cmult(task1_lab_U * task1_lab_phi_cache[idx],
                                                              task2_lab_U * task2_lab_phi_cache[idx]);
                        task1task2_lab_phi_flag[idx] = true;
                    }
                    if (!task1task2_lab_psi_flag[task1_role * TASK2_LABEL_SIZE + task2_role]) {
                        task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role]
                                = cmult(pick(task1_lab_psi_cache, task1_role),
                                        pick(task2_lab_psi_cache, task2_role));
                        task1task2_lab_psi_flag[task1_role * TASK2_LABEL_SIZE + task2_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task1task2_lab_phi_cache[idx],
                                  task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role])));
                } else if (task1_role >= 0 && task2_role < 0 && task3_role >= 0) {
                    if (!task1task3_lab_phi_flag[idx]) {
                        task1task3_lab_phi_cache[idx] = cmult(task1_lab_U * task1_lab_phi_cache[idx],
                                                              task3_lab_U * task3_lab_phi_cache[idx]);
                        task1task3_lab_phi_flag[idx] = true;
                    }
                    if (!task1task3_lab_psi_flag[task1_role * TASK3_LABEL_SIZE + task3_role]) {
                        task1task3_lab_psi_cache[task1_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(pick(task1_lab_psi_cache, task1_role),
                                        pick(task3_lab_psi_cache, task3_role));
                        task1task3_lab_psi_flag[task1_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task1task3_lab_phi_cache[idx],
                                  task1task3_lab_psi_cache[task1_role * TASK3_LABEL_SIZE + task3_role])));
                } else if (task1_role < 0 && task2_role >= 0 && task3_role >= 0) {
                    if (!task2task3_lab_phi_flag[idx]) {
                        task2task3_lab_phi_cache[idx] = cmult(task2_lab_U * task2_lab_phi_cache[idx],
                                                              task3_lab_U * task3_lab_phi_cache[idx]);
                        task2task3_lab_phi_flag[idx] = true;
                    }
                    if (!task2task3_lab_psi_flag[task2_role * TASK3_LABEL_SIZE + task3_role]) {
                        task2task3_lab_psi_cache[task2_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(pick(task2_lab_psi_cache, task2_role),
                                        pick(task3_lab_psi_cache, task3_role));
                        task2task3_lab_psi_flag[task2_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task2task3_lab_phi_cache[idx],
                                  task2task3_lab_psi_cache[task2_role * TASK3_LABEL_SIZE + task3_role])));
                } else if (task1_role >= 0 && task2_role >= 0 && task3_role >= 0) {
                    if (!all_lab_phi_flag[idx]) {
                        all_lab_phi_cache[idx] = cmult(task1task2_lab_phi_cache[idx],
                                                       task3_lab_U * task3_lab_phi_cache[idx]);
                        all_lab_phi_flag[idx] = true;
                    }
                    if (!all_lab_psi_flag[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                          + task2_role * TASK3_LABEL_SIZE + task3_role]) {
                        all_lab_psi_cache[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                          + task2_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role],
                                        pick(task3_lab_psi_cache, task3_role));
                        all_lab_psi_flag[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                         + task2_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(all_lab_phi_cache[idx],
                                  all_lab_psi_cache[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                                    + task2_role * TASK3_LABEL_SIZE + task3_role])));
                } else {
                    CHECK(false) << "Cross form labeled arc error. Giving up.";
                }
                ctf_scores[r] = dynet::as_scalar(cg.incremental_forward(ctf_exps[r]));
            } else {
                CHECK(false) << "part type mistake: " << (*ctf_parts)[r]->type() << endl;
            }
        }
        /* ctf end */

        SemanticDecoder *semantic_decoder = static_cast <SemanticDecoder *> (decoder_);
        if (!is_train) {
            semantic_decoder->DecodeCrossForm(instance,
                                              task1_parts, task1_scores, &task1_predicted_outputs,
                                              task2_parts, task2_scores, &task2_predicted_outputs,
                                              task3_parts, task3_scores, &task3_predicted_outputs,
                                              ctf_parts, ctf_scores, &ctf_predicted_outputs);

            for (int r = 0; r < task1_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            for (int r = 0; r < task2_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < task3_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < ctf_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(ctf_gold_outputs[r], ctf_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (ctf_predicted_outputs[r] - ctf_gold_outputs[r]) * ctf_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            Expression loss = Expression(&cg, cg.add_input(0.0));
            if (i_errs.size() > 0) {
                loss = loss + sum(i_errs);
            }
            return loss;
        }

        double s_loss = 0.0, s_cost = 0.0, cost = 0.0;
        semantic_decoder->DecodeCostAugmentedCrossForm(instance,
                                                       task1_parts, task1_scores, task1_gold_outputs,
                                                       &task1_predicted_outputs,
                                                       task2_parts, task2_scores, task2_gold_outputs,
                                                       &task2_predicted_outputs,
                                                       task3_parts, task3_scores, task3_gold_outputs,
                                                       &task3_predicted_outputs,
                                                       ctf_parts, ctf_scores, ctf_gold_outputs,
                                                       &ctf_predicted_outputs,
                                                       &s_cost, &s_loss);
        cost += s_cost;
        for (int r = 0; r < task1_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task2_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task3_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < ctf_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(ctf_gold_outputs[r], ctf_predicted_outputs[r], 1e-6)) {
                Expression i_err = (ctf_predicted_outputs[r] - ctf_gold_outputs[r]) * ctf_exps[r];
                i_errs.push_back(i_err);
            }
        }
        Expression loss = Expression(&cg, cg.add_input(cost));
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};

template<class Builder>
class Freda1 : public biLSTM {
private:
    // task1
    // predicate part
    dynet::Parameter task1_p_pred_w1;
    dynet::Parameter task1_p_pred_b1;
    dynet::Parameter task1_p_pred_w2;
    dynet::Parameter task1_p_pred_b2;
    dynet::Parameter task1_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task1_p_unlab_w1_pred;
    dynet::Parameter task1_p_unlab_w1_arg;
    dynet::Parameter task1_p_unlab_b1;
    dynet::Parameter task1_p_unlab_w2;
    dynet::Parameter task1_p_unlab_b2;
    dynet::Parameter task1_p_unlab_w3;
    // labeled arc
    dynet::Parameter task1_p_lab_w1_pred;
    dynet::Parameter task1_p_lab_w1_arg;
    dynet::Parameter task1_p_lab_b1;
    dynet::Parameter task1_p_lab_w2;
    dynet::Parameter task1_p_lab_b2;
    dynet::Parameter task1_p_lab_w3;
    dynet::Parameter task1_p_lab_b3;

    // task2
    // predicate part
    dynet::Parameter task2_p_pred_w1;
    dynet::Parameter task2_p_pred_b1;
    dynet::Parameter task2_p_pred_w2;
    dynet::Parameter task2_p_pred_b2;
    dynet::Parameter task2_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task2_p_unlab_w1_pred;
    dynet::Parameter task2_p_unlab_w1_arg;
    dynet::Parameter task2_p_unlab_b1;
    dynet::Parameter task2_p_unlab_w2;
    dynet::Parameter task2_p_unlab_b2;
    dynet::Parameter task2_p_unlab_w3;
    // labeled arc
    dynet::Parameter task2_p_lab_w1_pred;
    dynet::Parameter task2_p_lab_w1_arg;
    dynet::Parameter task2_p_lab_b1;
    dynet::Parameter task2_p_lab_w2;
    dynet::Parameter task2_p_lab_b2;
    dynet::Parameter task2_p_lab_w3;
    dynet::Parameter task2_p_lab_b3;

    // task3
    // predicate part
    dynet::Parameter task3_p_pred_w1;
    dynet::Parameter task3_p_pred_b1;
    dynet::Parameter task3_p_pred_w2;
    dynet::Parameter task3_p_pred_b2;
    dynet::Parameter task3_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task3_p_unlab_w1_pred;
    dynet::Parameter task3_p_unlab_w1_arg;
    dynet::Parameter task3_p_unlab_b1;
    dynet::Parameter task3_p_unlab_w2;
    dynet::Parameter task3_p_unlab_b2;
    dynet::Parameter task3_p_unlab_w3;
    // labeled arc
    dynet::Parameter task3_p_lab_w1_pred;
    dynet::Parameter task3_p_lab_w1_arg;
    dynet::Parameter task3_p_lab_b1;
    dynet::Parameter task3_p_lab_w2;
    dynet::Parameter task3_p_lab_b2;
    dynet::Parameter task3_p_lab_w3;
    dynet::Parameter task3_p_lab_b3;

    unsigned TASK1_LABEL_SIZE;
    unsigned TASK2_LABEL_SIZE;
    unsigned TASK3_LABEL_SIZE;

    Builder task1_l2rbuilder;
    Builder task1_r2lbuilder;
    Builder task2_l2rbuilder;
    Builder task2_r2lbuilder;
    Builder task3_l2rbuilder;
    Builder task3_r2lbuilder;
    Builder shared_l2rbuilder;
    Builder shared_r2lbuilder;

public:
    explicit Freda1(dynet::Model *model) {
    }

    explicit Freda1(SemanticOptions *semantic_option, const int &task1_num_roles, const int &task2_num_roles,
                    const int &task3_num_roles, dynet::Model *model) :
            task1_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task1_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task2_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task2_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task3_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task3_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            shared_l2rbuilder(semantic_option->num_lstm_layers(),
                              semantic_option->pre_word_dim() + semantic_option->word_dim() +
                              semantic_option->pos_dim(),
                              semantic_option->lstm_dim(), *model),
            shared_r2lbuilder(semantic_option->num_lstm_layers(),
                              semantic_option->pre_word_dim() + semantic_option->word_dim() +
                              semantic_option->pos_dim(),
                              semantic_option->lstm_dim(), *model) {
        LAYERS = semantic_option->num_lstm_layers();
        PRE_WORD_DIM = semantic_option->pre_word_dim();
        WORD_DIM = semantic_option->word_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->lstm_dim();
        MLP_DIM = semantic_option->mlp_dim();
        TASK1_LABEL_SIZE = task1_num_roles;
        TASK2_LABEL_SIZE = task2_num_roles;
        TASK3_LABEL_SIZE = task3_num_roles;
    }

    void InitParams(dynet::Model *model) {
        // shared
        p_embed_pre_word = model->add_lookup_parameters(VOCAB_SIZE, {PRE_WORD_DIM});
        p_embed_word = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        p_embed_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // task1
        // predicate part
        task1_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_pred_b1 = model->add_parameters({MLP_DIM});
        task1_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_pred_b2 = model->add_parameters({MLP_DIM});
        task1_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task1_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task1_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_lab_b1 = model->add_parameters({MLP_DIM});
        task1_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_lab_b2 = model->add_parameters({MLP_DIM});
        task1_p_lab_w3 = model->add_parameters({TASK1_LABEL_SIZE, MLP_DIM});
        task1_p_lab_b3 = model->add_parameters({TASK1_LABEL_SIZE});

        // task2
        // predicate part
        task2_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_pred_b1 = model->add_parameters({MLP_DIM});
        task2_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_pred_b2 = model->add_parameters({MLP_DIM});
        task2_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task2_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task2_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_lab_b1 = model->add_parameters({MLP_DIM});
        task2_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_lab_b2 = model->add_parameters({MLP_DIM});
        task2_p_lab_w3 = model->add_parameters({TASK2_LABEL_SIZE, MLP_DIM});
        task2_p_lab_b3 = model->add_parameters({TASK2_LABEL_SIZE});

        // task3
        // predicate part
        task3_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_pred_b1 = model->add_parameters({MLP_DIM});
        task3_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_pred_b2 = model->add_parameters({MLP_DIM});
        task3_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task3_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task3_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_lab_b1 = model->add_parameters({MLP_DIM});
        task3_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_lab_b2 = model->add_parameters({MLP_DIM});
        task3_p_lab_w3 = model->add_parameters({TASK3_LABEL_SIZE, MLP_DIM});
        task3_p_lab_b3 = model->add_parameters({TASK3_LABEL_SIZE});
    }

    Expression BuildGraph(Instance *instance, Decoder *decoder_, const bool &is_train,
                          Parts *task1_parts, vector<double> &task1_scores, const vector<double> &task1_gold_outputs,
                          vector<double> &task1_predicted_outputs,
                          Parts *task2_parts, vector<double> &task2_scores, const vector<double> &task2_gold_outputs,
                          vector<double> &task2_predicted_outputs,
                          Parts *task3_parts, vector<double> &task3_scores, const vector<double> &task3_gold_outputs,
                          vector<double> &task3_predicted_outputs,
                          Parts *ctf_parts, vector<double> &ctf_scores,
                          const vector<double> &ctf_gold_outputs,
                          vector<double> &ctf_predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count, dynet::ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> pos = sentence->GetPosIds();
        task1_l2rbuilder.new_graph(cg);
        task1_l2rbuilder.start_new_sequence();
        task1_r2lbuilder.new_graph(cg);
        task1_r2lbuilder.start_new_sequence();

        task2_l2rbuilder.new_graph(cg);
        task2_l2rbuilder.start_new_sequence();
        task2_r2lbuilder.new_graph(cg);
        task2_r2lbuilder.start_new_sequence();

        task3_l2rbuilder.new_graph(cg);
        task3_l2rbuilder.start_new_sequence();
        task3_r2lbuilder.new_graph(cg);
        task3_r2lbuilder.start_new_sequence();

        shared_l2rbuilder.new_graph(cg);
        shared_l2rbuilder.start_new_sequence();
        shared_r2lbuilder.new_graph(cg);
        shared_r2lbuilder.start_new_sequence();

        // task1
        // predicate part
        Expression task1_pred_w1 = parameter(cg, task1_p_pred_w1);
        Expression task1_pred_b1 = parameter(cg, task1_p_pred_b1);
        Expression task1_pred_w2 = parameter(cg, task1_p_pred_w2);
        Expression task1_pred_b2 = parameter(cg, task1_p_pred_b2);
        Expression task1_pred_w3 = parameter(cg, task1_p_pred_w3);
        // unlabeled arc
        Expression task1_unlab_w1_pred = parameter(cg, task1_p_unlab_w1_pred);
        Expression task1_unlab_w1_arg = parameter(cg, task1_p_unlab_w1_arg);
        Expression task1_unlab_b1 = parameter(cg, task1_p_unlab_b1);
        Expression task1_unlab_w2 = parameter(cg, task1_p_unlab_w2);
        Expression task1_unlab_b2 = parameter(cg, task1_p_unlab_b2);
        Expression task1_unlab_w3 = parameter(cg, task1_p_unlab_w3);
        // labeled arc
        Expression task1_lab_w1_pred = parameter(cg, task1_p_lab_w1_pred);
        Expression task1_lab_w1_arg = parameter(cg, task1_p_lab_w1_arg);
        Expression task1_lab_b1 = parameter(cg, task1_p_lab_b1);
        Expression task1_lab_w2 = parameter(cg, task1_p_lab_w2);
        Expression task1_lab_b2 = parameter(cg, task1_p_lab_b2);
        Expression task1_lab_w3 = parameter(cg, task1_p_lab_w3);
        Expression task1_lab_b3 = parameter(cg, task1_p_lab_b3);

        // task2
        // predicate part
        Expression task2_pred_w1 = parameter(cg, task2_p_pred_w1);
        Expression task2_pred_b1 = parameter(cg, task2_p_pred_b1);
        Expression task2_pred_w2 = parameter(cg, task2_p_pred_w2);
        Expression task2_pred_b2 = parameter(cg, task2_p_pred_b2);
        Expression task2_pred_w3 = parameter(cg, task2_p_pred_w3);
        // unlabeled arc
        Expression task2_unlab_w1_pred = parameter(cg, task2_p_unlab_w1_pred);
        Expression task2_unlab_w1_arg = parameter(cg, task2_p_unlab_w1_arg);
        Expression task2_unlab_b1 = parameter(cg, task2_p_unlab_b1);
        Expression task2_unlab_w2 = parameter(cg, task2_p_unlab_w2);
        Expression task2_unlab_b2 = parameter(cg, task2_p_unlab_b2);
        Expression task2_unlab_w3 = parameter(cg, task2_p_unlab_w3);
        // labeled arc
        Expression task2_lab_w1_pred = parameter(cg, task2_p_lab_w1_pred);
        Expression task2_lab_w1_arg = parameter(cg, task2_p_lab_w1_arg);
        Expression task2_lab_b1 = parameter(cg, task2_p_lab_b1);
        Expression task2_lab_w2 = parameter(cg, task2_p_lab_w2);
        Expression task2_lab_b2 = parameter(cg, task2_p_lab_b2);
        Expression task2_lab_w3 = parameter(cg, task2_p_lab_w3);
        Expression task2_lab_b3 = parameter(cg, task2_p_lab_b3);

        // task3
        // predicate part
        Expression task3_pred_w1 = parameter(cg, task3_p_pred_w1);
        Expression task3_pred_b1 = parameter(cg, task3_p_pred_b1);
        Expression task3_pred_w2 = parameter(cg, task3_p_pred_w2);
        Expression task3_pred_b2 = parameter(cg, task3_p_pred_b2);
        Expression task3_pred_w3 = parameter(cg, task3_p_pred_w3);
        // unlabeled arc
        Expression task3_unlab_w1_pred = parameter(cg, task3_p_unlab_w1_pred);
        Expression task3_unlab_w1_arg = parameter(cg, task3_p_unlab_w1_arg);
        Expression task3_unlab_b1 = parameter(cg, task3_p_unlab_b1);
        Expression task3_unlab_w2 = parameter(cg, task3_p_unlab_w2);
        Expression task3_unlab_b2 = parameter(cg, task3_p_unlab_b2);
        Expression task3_unlab_w3 = parameter(cg, task3_p_unlab_w3);
        // labeled arc
        Expression task3_lab_w1_pred = parameter(cg, task3_p_lab_w1_pred);
        Expression task3_lab_w1_arg = parameter(cg, task3_p_lab_w1_arg);
        Expression task3_lab_b1 = parameter(cg, task3_p_lab_b1);
        Expression task3_lab_w2 = parameter(cg, task3_p_lab_w2);
        Expression task3_lab_b2 = parameter(cg, task3_p_lab_b2);
        Expression task3_lab_w3 = parameter(cg, task3_p_lab_w3);
        Expression task3_lab_b3 = parameter(cg, task3_p_lab_b3);

        vector<Expression> ex_words(slen), i_errs;
        vector<Expression> task1_ex_l2r(slen), task1_ex_r2l(slen);
        vector<Expression> task2_ex_l2r(slen), task2_ex_r2l(slen);
        vector<Expression> task3_ex_l2r(slen), task3_ex_r2l(slen);
        vector<Expression> shared_ex_l2r(slen), shared_ex_r2l(slen);
        for (int i = 0; i < slen; ++i) {
            int word_idx = words[i];
            if (use_word_dropout && word_idx != UNK_ID) {
                int count = form_count->find(word_idx)->second;
                float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                if (rand_float < word_dropout_rate / (static_cast<float> (count) + word_dropout_rate))
                    word_idx = UNK_ID;
            }
            Expression x_pre_word = lookup(cg, p_embed_pre_word, word_idx);
            Expression x_word = lookup(cg, p_embed_word, word_idx);
            Expression x_pos = lookup(cg, p_embed_pos, pos[i]);
            ex_words[i] = concatenate({x_pre_word, x_word, x_pos});
            task1_ex_l2r[i] = task1_l2rbuilder.add_input(ex_words[i]);
            task2_ex_l2r[i] = task2_l2rbuilder.add_input(ex_words[i]);
            task3_ex_l2r[i] = task3_l2rbuilder.add_input(ex_words[i]);
            shared_ex_l2r[i] = shared_l2rbuilder.add_input(ex_words[i]);
        }
        for (int i = 0; i < slen; ++i) {
            task1_ex_r2l[slen - i - 1] = task1_r2lbuilder.add_input(ex_words[slen - i - 1]);
            task2_ex_r2l[slen - i - 1] = task2_r2lbuilder.add_input(ex_words[slen - i - 1]);
            task3_ex_r2l[slen - i - 1] = task3_r2lbuilder.add_input(ex_words[slen - i - 1]);
            shared_ex_r2l[slen - i - 1] = shared_r2lbuilder.add_input(ex_words[slen - i - 1]);
        }
        vector<Expression> task1_unlab_pred_exs, task1_unlab_arg_exs;
        vector<Expression> task1_lab_pred_exs, task1_lab_arg_exs;
        vector<Expression> task2_unlab_pred_exs, task2_unlab_arg_exs;
        vector<Expression> task2_lab_pred_exs, task2_lab_arg_exs;
        vector<Expression> task3_unlab_pred_exs, task3_unlab_arg_exs;
        vector<Expression> task3_lab_pred_exs, task3_lab_arg_exs;

        for (int i = 0; i < slen; ++i) {
            Expression task1_word_ex = concatenate(
                    {task1_ex_l2r[i], task1_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task1_unlab_pred_exs.push_back(task1_unlab_w1_pred * task1_word_ex);
            task1_unlab_arg_exs.push_back(task1_unlab_w1_arg * task1_word_ex);
            task1_lab_pred_exs.push_back(task1_lab_w1_pred * task1_word_ex);
            task1_lab_arg_exs.push_back(task1_lab_w1_arg * task1_word_ex);

            Expression task2_word_ex = concatenate(
                    {task2_ex_l2r[i], task2_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task2_unlab_pred_exs.push_back(task2_unlab_w1_pred * task2_word_ex);
            task2_unlab_arg_exs.push_back(task2_unlab_w1_arg * task2_word_ex);
            task2_lab_pred_exs.push_back(task2_lab_w1_pred * task2_word_ex);
            task2_lab_arg_exs.push_back(task2_lab_w1_arg * task2_word_ex);

            Expression task3_word_ex = concatenate(
                    {task3_ex_l2r[i], task3_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task3_unlab_pred_exs.push_back(task3_unlab_w1_pred * task3_word_ex);
            task3_unlab_arg_exs.push_back(task3_unlab_w1_arg * task3_word_ex);
            task3_lab_pred_exs.push_back(task3_lab_w1_pred * task3_word_ex);
            task3_lab_arg_exs.push_back(task3_lab_w1_arg * task3_word_ex);
        }
        /* task1 begin */
        vector<Expression> task1_exps(task1_parts->size());
        task1_scores.assign(task1_parts->size(), 0.0);
        task1_predicted_outputs.assign(task1_parts->size(), 0.0);
        SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
        for (int r = 0; r < task1_parts->size(); ++r) {
            if ((*task1_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task1_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task1_pred_ex = concatenate({task1_ex_l2r[idx_pred], task1_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task1_pred_MLP_in = tanh(task1_pred_w1 * task1_pred_ex + task1_pred_b1);
                Expression task1_pred_phi = tanh(task1_pred_w2 * task1_pred_MLP_in + task1_pred_b2);
                task1_exps[r] = task1_pred_w3 * task1_pred_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
            } else if ((*task1_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task1_unlab_MLP_in = tanh(
                        task1_unlab_pred_exs[idx_pred] + task1_unlab_arg_exs[idx_arg] + task1_unlab_b1);
                Expression task1_lab_MLP_in = tanh(
                        task1_lab_pred_exs[idx_pred] + task1_lab_arg_exs[idx_arg] + task1_lab_b1);
                Expression task1_unlab_phi = tanh(task1_unlab_w2 * task1_unlab_MLP_in + task1_unlab_b2);
                Expression task1_lab_phi = tanh(task1_lab_w2 * task1_lab_MLP_in + task1_lab_b2);
                task1_exps[r] = task1_unlab_w3 * task1_unlab_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
                Expression lab_MLP_out = task1_lab_w3 * task1_lab_phi + task1_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task1_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task1_parts->size());
                    CHECK_EQ((*task1_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task1_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task1_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task1_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task1 end */
        /* task2 begin */
        vector<Expression> task2_exps(task2_parts->size());
        task2_scores.assign(task2_parts->size(), 0.0);
        task2_predicted_outputs.assign(task2_parts->size(), 0.0);
        SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
        for (int r = 0; r < task2_parts->size(); ++r) {
            if ((*task2_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task2_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task2_pred_ex = concatenate({task2_ex_l2r[idx_pred], task2_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task2_pred_MLP_in = tanh(task2_pred_w1 * task2_pred_ex + task2_pred_b1);
                Expression task2_pred_phi = tanh(task2_pred_w2 * task2_pred_MLP_in + task2_pred_b2);
                task2_exps[r] = task2_pred_w3 * task2_pred_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
            } else if ((*task2_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task2_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task2_unlab_MLP_in = tanh(
                        task2_unlab_pred_exs[idx_pred] + task2_unlab_arg_exs[idx_arg] + task2_unlab_b1);
                Expression task2_lab_MLP_in = tanh(
                        task2_lab_pred_exs[idx_pred] + task2_lab_arg_exs[idx_arg] + task2_lab_b1);
                Expression task2_unlab_phi = tanh(task2_unlab_w2 * task2_unlab_MLP_in + task2_unlab_b2);
                Expression task2_lab_phi = tanh(task2_lab_w2 * task2_lab_MLP_in + task2_lab_b2);
                task2_exps[r] = task2_unlab_w3 * task2_unlab_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
                Expression lab_MLP_out = task2_lab_w3 * task2_lab_phi + task2_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task2_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task2_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task2_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task2 end */
        /* task3 begin */
        vector<Expression> task3_exps(task3_parts->size());
        task3_scores.assign(task3_parts->size(), 0.0);
        task3_predicted_outputs.assign(task3_parts->size(), 0.0);
        SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
        for (int r = 0; r < task3_parts->size(); ++r) {
            if ((*task3_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task3_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task3_pred_ex = concatenate({task3_ex_l2r[idx_pred], task3_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task3_pred_MLP_in = tanh(task3_pred_w1 * task3_pred_ex + task3_pred_b1);
                Expression task3_pred_phi = tanh(task3_pred_w2 * task3_pred_MLP_in + task3_pred_b2);
                task3_exps[r] = task3_pred_w3 * task3_pred_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
            } else if ((*task3_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task3_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task3_unlab_MLP_in = tanh(
                        task3_unlab_pred_exs[idx_pred] + task3_unlab_arg_exs[idx_arg] + task3_unlab_b1);
                Expression task3_lab_MLP_in = tanh(
                        task3_lab_pred_exs[idx_pred] + task3_lab_arg_exs[idx_arg] + task3_lab_b1);
                Expression task3_unlab_phi = tanh(task3_unlab_w2 * task3_unlab_MLP_in + task3_unlab_b2);
                Expression task3_lab_phi = tanh(task3_lab_w2 * task3_lab_MLP_in + task3_lab_b2);
                task3_exps[r] = task3_unlab_w3 * task3_unlab_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
                Expression lab_MLP_out = task3_lab_w3 * task3_lab_phi + task3_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task3_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task3_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task3_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task3 end */
        if (!is_train) {
            decoder_->Decode(instance, task1_parts, task1_scores, &task1_predicted_outputs);
            decoder_->Decode(instance, task2_parts, task2_scores, &task2_predicted_outputs);
            decoder_->Decode(instance, task3_parts, task3_scores, &task3_predicted_outputs);
            for (int r = 0; r < task1_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            for (int r = 0; r < task2_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < task3_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            Expression loss = Expression(&cg, cg.add_input(0.0));
            if (i_errs.size() > 0) {
                loss = loss + sum(i_errs);
            }
            return loss;
        }
        double s_loss = 0.0, s_cost = 0.0, cost = 0.0;
        decoder_->DecodeCostAugmented(instance, task1_parts, task1_scores, task1_gold_outputs,
                                      &task1_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;
        decoder_->DecodeCostAugmented(instance, task2_parts, task2_scores, task2_gold_outputs,
                                      &task2_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;
        decoder_->DecodeCostAugmented(instance, task3_parts, task3_scores, task3_gold_outputs,
                                      &task3_predicted_outputs, &s_cost, &s_loss);
        cost += s_cost;
        for (int r = 0; r < task1_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task2_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task3_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                i_errs.push_back(i_err);
            }
        }
        Expression loss = Expression(&cg, cg.add_input(cost));
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};

template<class Builder>
class Freda3 : public biLSTM {
private:
    // task1
    // predicate part
    dynet::Parameter task1_p_pred_w1;
    dynet::Parameter task1_p_pred_b1;
    dynet::Parameter task1_p_pred_w2;
    dynet::Parameter task1_p_pred_b2;
    dynet::Parameter task1_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task1_p_unlab_w1_pred;
    dynet::Parameter task1_p_unlab_w1_arg;
    dynet::Parameter task1_p_unlab_b1;
    dynet::Parameter task1_p_unlab_w2;
    dynet::Parameter task1_p_unlab_b2;
    dynet::Parameter task1_p_unlab_w3;
    // labeled arc
    dynet::Parameter task1_p_lab_w1_pred;
    dynet::Parameter task1_p_lab_w1_arg;
    dynet::Parameter task1_p_lab_b1;
    dynet::Parameter task1_p_lab_w2;
    dynet::Parameter task1_p_lab_b2;
    dynet::Parameter task1_p_lab_w3;
    dynet::Parameter task1_p_lab_b3;

    // task2
    // predicate part
    dynet::Parameter task2_p_pred_w1;
    dynet::Parameter task2_p_pred_b1;
    dynet::Parameter task2_p_pred_w2;
    dynet::Parameter task2_p_pred_b2;
    dynet::Parameter task2_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task2_p_unlab_w1_pred;
    dynet::Parameter task2_p_unlab_w1_arg;
    dynet::Parameter task2_p_unlab_b1;
    dynet::Parameter task2_p_unlab_w2;
    dynet::Parameter task2_p_unlab_b2;
    dynet::Parameter task2_p_unlab_w3;
    // labeled arc
    dynet::Parameter task2_p_lab_w1_pred;
    dynet::Parameter task2_p_lab_w1_arg;
    dynet::Parameter task2_p_lab_b1;
    dynet::Parameter task2_p_lab_w2;
    dynet::Parameter task2_p_lab_b2;
    dynet::Parameter task2_p_lab_w3;
    dynet::Parameter task2_p_lab_b3;

    // task3
    // predicate part
    dynet::Parameter task3_p_pred_w1;
    dynet::Parameter task3_p_pred_b1;
    dynet::Parameter task3_p_pred_w2;
    dynet::Parameter task3_p_pred_b2;
    dynet::Parameter task3_p_pred_w3;
    // unlabeled arc
    dynet::Parameter task3_p_unlab_w1_pred;
    dynet::Parameter task3_p_unlab_w1_arg;
    dynet::Parameter task3_p_unlab_b1;
    dynet::Parameter task3_p_unlab_w2;
    dynet::Parameter task3_p_unlab_b2;
    dynet::Parameter task3_p_unlab_w3;
    // labeled arc
    dynet::Parameter task3_p_lab_w1_pred;
    dynet::Parameter task3_p_lab_w1_arg;
    dynet::Parameter task3_p_lab_b1;
    dynet::Parameter task3_p_lab_w2;
    dynet::Parameter task3_p_lab_b2;
    dynet::Parameter task3_p_lab_w3;
    dynet::Parameter task3_p_lab_b3;

    // ctf
    // unlabeled
    dynet::Parameter task1_p_unlab_U;
    dynet::Parameter task2_p_unlab_U;
    dynet::Parameter task3_p_unlab_U;
    dynet::Parameter task1_p_unlab_V;
    dynet::Parameter task2_p_unlab_V;
    dynet::Parameter task3_p_unlab_V;
    // labeled
    dynet::Parameter task1_p_lab_U;
    dynet::Parameter task2_p_lab_U;
    dynet::Parameter task3_p_lab_U;
    dynet::Parameter task1_p_lab_V;
    dynet::Parameter task2_p_lab_V;
    dynet::Parameter task3_p_lab_V;

    unsigned TASK1_LABEL_SIZE;
    unsigned TASK2_LABEL_SIZE;
    unsigned TASK3_LABEL_SIZE;
    unsigned RANK;

    Builder task1_l2rbuilder;
    Builder task1_r2lbuilder;
    Builder task2_l2rbuilder;
    Builder task2_r2lbuilder;
    Builder task3_l2rbuilder;
    Builder task3_r2lbuilder;
    Builder shared_l2rbuilder;
    Builder shared_r2lbuilder;

public:
    explicit Freda3(dynet::Model *model) {
    }

    explicit Freda3(SemanticOptions *semantic_option, const int &task1_num_roles, const int &task2_num_roles,
                    const int &task3_num_roles, dynet::Model *model) :
            task1_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task1_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task2_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task2_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task3_l2rbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            task3_r2lbuilder(semantic_option->num_lstm_layers(),
                             semantic_option->pre_word_dim() + semantic_option->word_dim() + semantic_option->pos_dim(),
                             semantic_option->lstm_dim(), *model),
            shared_l2rbuilder(semantic_option->num_lstm_layers(),
                              semantic_option->pre_word_dim() + semantic_option->word_dim() +
                              semantic_option->pos_dim(),
                              semantic_option->lstm_dim(), *model),
            shared_r2lbuilder(semantic_option->num_lstm_layers(),
                              semantic_option->pre_word_dim() + semantic_option->word_dim() +
                              semantic_option->pos_dim(),
                              semantic_option->lstm_dim(), *model) {
        LAYERS = semantic_option->num_lstm_layers();
        PRE_WORD_DIM = semantic_option->pre_word_dim();
        WORD_DIM = semantic_option->word_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->lstm_dim();
        MLP_DIM = semantic_option->mlp_dim();
        TASK1_LABEL_SIZE = task1_num_roles;
        TASK2_LABEL_SIZE = task2_num_roles;
        TASK3_LABEL_SIZE = task3_num_roles;
        RANK = semantic_option->rank();
    }

    void InitParams(dynet::Model *model) {
        // shared
        p_embed_pre_word = model->add_lookup_parameters(VOCAB_SIZE, {PRE_WORD_DIM});
        p_embed_word = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        p_embed_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // task1
        // predicate part
        task1_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_pred_b1 = model->add_parameters({MLP_DIM});
        task1_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_pred_b2 = model->add_parameters({MLP_DIM});
        task1_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task1_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task1_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task1_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task1_p_lab_b1 = model->add_parameters({MLP_DIM});
        task1_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task1_p_lab_b2 = model->add_parameters({MLP_DIM});
        task1_p_lab_w3 = model->add_parameters({TASK1_LABEL_SIZE, MLP_DIM});
        task1_p_lab_b3 = model->add_parameters({TASK1_LABEL_SIZE});

        // task2
        // predicate part
        task2_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_pred_b1 = model->add_parameters({MLP_DIM});
        task2_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_pred_b2 = model->add_parameters({MLP_DIM});
        task2_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task2_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task2_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task2_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task2_p_lab_b1 = model->add_parameters({MLP_DIM});
        task2_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task2_p_lab_b2 = model->add_parameters({MLP_DIM});
        task2_p_lab_w3 = model->add_parameters({TASK2_LABEL_SIZE, MLP_DIM});
        task2_p_lab_b3 = model->add_parameters({TASK2_LABEL_SIZE});

        // task3
        // predicate part
        task3_p_pred_w1 = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_pred_b1 = model->add_parameters({MLP_DIM});
        task3_p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_pred_b2 = model->add_parameters({MLP_DIM});
        task3_p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        task3_p_unlab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_unlab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_unlab_b1 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_unlab_b2 = model->add_parameters({MLP_DIM});
        task3_p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        task3_p_lab_w1_pred = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_lab_w1_arg = model->add_parameters({MLP_DIM, 4 * LSTM_DIM});
        task3_p_lab_b1 = model->add_parameters({MLP_DIM});
        task3_p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        task3_p_lab_b2 = model->add_parameters({MLP_DIM});
        task3_p_lab_w3 = model->add_parameters({TASK3_LABEL_SIZE, MLP_DIM});
        task3_p_lab_b3 = model->add_parameters({TASK3_LABEL_SIZE});

        // ctf
        // unlabeled
        task1_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task2_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task3_p_unlab_U = model->add_parameters({RANK, MLP_DIM});
        task1_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        task2_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        task3_p_unlab_V = model->add_parameters({RANK, MLP_DIM});
        // labeled
        task1_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task2_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task3_p_lab_U = model->add_parameters({RANK, MLP_DIM});
        task1_p_lab_V = model->add_parameters({RANK, MLP_DIM});
        task2_p_lab_V = model->add_parameters({RANK, MLP_DIM});
        task3_p_lab_V = model->add_parameters({RANK, MLP_DIM});
    }

    Expression BuildGraph(Instance *instance, Decoder *decoder_, const bool &is_train,
                          Parts *task1_parts, vector<double> &task1_scores, const vector<double> &task1_gold_outputs,
                          vector<double> &task1_predicted_outputs,
                          Parts *task2_parts, vector<double> &task2_scores, const vector<double> &task2_gold_outputs,
                          vector<double> &task2_predicted_outputs,
                          Parts *task3_parts, vector<double> &task3_scores, const vector<double> &task3_gold_outputs,
                          vector<double> &task3_predicted_outputs,
                          Parts *ctf_parts, vector<double> &ctf_scores, const vector<double> &ctf_gold_outputs,
                          vector<double> &ctf_predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count, dynet::ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> pos = sentence->GetPosIds();
        task1_l2rbuilder.new_graph(cg);
        task1_l2rbuilder.start_new_sequence();
        task1_r2lbuilder.new_graph(cg);
        task1_r2lbuilder.start_new_sequence();

        task2_l2rbuilder.new_graph(cg);
        task2_l2rbuilder.start_new_sequence();
        task2_r2lbuilder.new_graph(cg);
        task2_r2lbuilder.start_new_sequence();

        task3_l2rbuilder.new_graph(cg);
        task3_l2rbuilder.start_new_sequence();
        task3_r2lbuilder.new_graph(cg);
        task3_r2lbuilder.start_new_sequence();

        shared_l2rbuilder.new_graph(cg);
        shared_l2rbuilder.start_new_sequence();
        shared_r2lbuilder.new_graph(cg);
        shared_r2lbuilder.start_new_sequence();

        // task1
        // predicate part
        Expression task1_pred_w1 = parameter(cg, task1_p_pred_w1);
        Expression task1_pred_b1 = parameter(cg, task1_p_pred_b1);
        Expression task1_pred_w2 = parameter(cg, task1_p_pred_w2);
        Expression task1_pred_b2 = parameter(cg, task1_p_pred_b2);
        Expression task1_pred_w3 = parameter(cg, task1_p_pred_w3);
        // unlabeled arc
        Expression task1_unlab_w1_pred = parameter(cg, task1_p_unlab_w1_pred);
        Expression task1_unlab_w1_arg = parameter(cg, task1_p_unlab_w1_arg);
        Expression task1_unlab_b1 = parameter(cg, task1_p_unlab_b1);
        Expression task1_unlab_w2 = parameter(cg, task1_p_unlab_w2);
        Expression task1_unlab_b2 = parameter(cg, task1_p_unlab_b2);
        Expression task1_unlab_w3 = parameter(cg, task1_p_unlab_w3);
        // labeled arc
        Expression task1_lab_w1_pred = parameter(cg, task1_p_lab_w1_pred);
        Expression task1_lab_w1_arg = parameter(cg, task1_p_lab_w1_arg);
        Expression task1_lab_b1 = parameter(cg, task1_p_lab_b1);
        Expression task1_lab_w2 = parameter(cg, task1_p_lab_w2);
        Expression task1_lab_b2 = parameter(cg, task1_p_lab_b2);
        Expression task1_lab_w3 = parameter(cg, task1_p_lab_w3);
        Expression task1_lab_b3 = parameter(cg, task1_p_lab_b3);

        // task2
        // predicate part
        Expression task2_pred_w1 = parameter(cg, task2_p_pred_w1);
        Expression task2_pred_b1 = parameter(cg, task2_p_pred_b1);
        Expression task2_pred_w2 = parameter(cg, task2_p_pred_w2);
        Expression task2_pred_b2 = parameter(cg, task2_p_pred_b2);
        Expression task2_pred_w3 = parameter(cg, task2_p_pred_w3);
        // unlabeled arc
        Expression task2_unlab_w1_pred = parameter(cg, task2_p_unlab_w1_pred);
        Expression task2_unlab_w1_arg = parameter(cg, task2_p_unlab_w1_arg);
        Expression task2_unlab_b1 = parameter(cg, task2_p_unlab_b1);
        Expression task2_unlab_w2 = parameter(cg, task2_p_unlab_w2);
        Expression task2_unlab_b2 = parameter(cg, task2_p_unlab_b2);
        Expression task2_unlab_w3 = parameter(cg, task2_p_unlab_w3);
        // labeled arc
        Expression task2_lab_w1_pred = parameter(cg, task2_p_lab_w1_pred);
        Expression task2_lab_w1_arg = parameter(cg, task2_p_lab_w1_arg);
        Expression task2_lab_b1 = parameter(cg, task2_p_lab_b1);
        Expression task2_lab_w2 = parameter(cg, task2_p_lab_w2);
        Expression task2_lab_b2 = parameter(cg, task2_p_lab_b2);
        Expression task2_lab_w3 = parameter(cg, task2_p_lab_w3);
        Expression task2_lab_b3 = parameter(cg, task2_p_lab_b3);

        // task3
        // predicate part
        Expression task3_pred_w1 = parameter(cg, task3_p_pred_w1);
        Expression task3_pred_b1 = parameter(cg, task3_p_pred_b1);
        Expression task3_pred_w2 = parameter(cg, task3_p_pred_w2);
        Expression task3_pred_b2 = parameter(cg, task3_p_pred_b2);
        Expression task3_pred_w3 = parameter(cg, task3_p_pred_w3);
        // unlabeled arc
        Expression task3_unlab_w1_pred = parameter(cg, task3_p_unlab_w1_pred);
        Expression task3_unlab_w1_arg = parameter(cg, task3_p_unlab_w1_arg);
        Expression task3_unlab_b1 = parameter(cg, task3_p_unlab_b1);
        Expression task3_unlab_w2 = parameter(cg, task3_p_unlab_w2);
        Expression task3_unlab_b2 = parameter(cg, task3_p_unlab_b2);
        Expression task3_unlab_w3 = parameter(cg, task3_p_unlab_w3);
        // labeled arc
        Expression task3_lab_w1_pred = parameter(cg, task3_p_lab_w1_pred);
        Expression task3_lab_w1_arg = parameter(cg, task3_p_lab_w1_arg);
        Expression task3_lab_b1 = parameter(cg, task3_p_lab_b1);
        Expression task3_lab_w2 = parameter(cg, task3_p_lab_w2);
        Expression task3_lab_b2 = parameter(cg, task3_p_lab_b2);
        Expression task3_lab_w3 = parameter(cg, task3_p_lab_w3);
        Expression task3_lab_b3 = parameter(cg, task3_p_lab_b3);

        // ctf
        Expression task1_unlab_U = parameter(cg, task1_p_unlab_U);
        Expression task2_unlab_U = parameter(cg, task2_p_unlab_U);
        Expression task3_unlab_U = parameter(cg, task3_p_unlab_U);
        Expression task1_unlab_V = parameter(cg, task1_p_unlab_V);
        Expression task2_unlab_V = parameter(cg, task2_p_unlab_V);
        Expression task3_unlab_V = parameter(cg, task3_p_unlab_V);
        Expression task1_lab_U = parameter(cg, task1_p_lab_U);
        Expression task2_lab_U = parameter(cg, task2_p_lab_U);
        Expression task3_lab_U = parameter(cg, task3_p_lab_U);
        Expression task1_lab_V = parameter(cg, task1_p_lab_V);
        Expression task2_lab_V = parameter(cg, task2_p_lab_V);
        Expression task3_lab_V = parameter(cg, task3_p_lab_V);

        vector<Expression> ex_words(slen), i_errs;
        vector<Expression> task1_ex_l2r(slen), task1_ex_r2l(slen);
        vector<Expression> task2_ex_l2r(slen), task2_ex_r2l(slen);
        vector<Expression> task3_ex_l2r(slen), task3_ex_r2l(slen);
        vector<Expression> shared_ex_l2r(slen), shared_ex_r2l(slen);
        for (int i = 0; i < slen; ++i) {
            int word_idx = words[i];
            if (use_word_dropout && word_idx != UNK_ID) {
                int count = form_count->find(word_idx)->second;
                float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                if (rand_float < word_dropout_rate / (static_cast<float> (count) + word_dropout_rate))
                    word_idx = UNK_ID;
            }
            Expression x_pre_word = lookup(cg, p_embed_pre_word, word_idx);
            Expression x_word = lookup(cg, p_embed_word, word_idx);
            Expression x_pos = lookup(cg, p_embed_pos, pos[i]);
            ex_words[i] = concatenate({x_pre_word, x_word, x_pos});
            task1_ex_l2r[i] = task1_l2rbuilder.add_input(ex_words[i]);
            task2_ex_l2r[i] = task2_l2rbuilder.add_input(ex_words[i]);
            task3_ex_l2r[i] = task3_l2rbuilder.add_input(ex_words[i]);
            shared_ex_l2r[i] = shared_l2rbuilder.add_input(ex_words[i]);
        }
        for (int i = 0; i < slen; ++i) {
            task1_ex_r2l[slen - i - 1] = task1_r2lbuilder.add_input(ex_words[slen - i - 1]);
            task2_ex_r2l[slen - i - 1] = task2_r2lbuilder.add_input(ex_words[slen - i - 1]);
            task3_ex_r2l[slen - i - 1] = task3_r2lbuilder.add_input(ex_words[slen - i - 1]);
            shared_ex_r2l[slen - i - 1] = shared_r2lbuilder.add_input(ex_words[slen - i - 1]);
        }
        vector<Expression> task1_unlab_pred_exs, task1_unlab_arg_exs;
        vector<Expression> task1_lab_pred_exs, task1_lab_arg_exs;
        vector<Expression> task2_unlab_pred_exs, task2_unlab_arg_exs;
        vector<Expression> task2_lab_pred_exs, task2_lab_arg_exs;
        vector<Expression> task3_unlab_pred_exs, task3_unlab_arg_exs;
        vector<Expression> task3_lab_pred_exs, task3_lab_arg_exs;

        for (int i = 0; i < slen; ++i) {
            Expression task1_word_ex = concatenate(
                    {task1_ex_l2r[i], task1_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task1_unlab_pred_exs.push_back(task1_unlab_w1_pred * task1_word_ex);
            task1_unlab_arg_exs.push_back(task1_unlab_w1_arg * task1_word_ex);
            task1_lab_pred_exs.push_back(task1_lab_w1_pred * task1_word_ex);
            task1_lab_arg_exs.push_back(task1_lab_w1_arg * task1_word_ex);

            Expression task2_word_ex = concatenate(
                    {task2_ex_l2r[i], task2_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task2_unlab_pred_exs.push_back(task2_unlab_w1_pred * task2_word_ex);
            task2_unlab_arg_exs.push_back(task2_unlab_w1_arg * task2_word_ex);
            task2_lab_pred_exs.push_back(task2_lab_w1_pred * task2_word_ex);
            task2_lab_arg_exs.push_back(task2_lab_w1_arg * task2_word_ex);

            Expression task3_word_ex = concatenate(
                    {task3_ex_l2r[i], task3_ex_r2l[i], shared_ex_l2r[i], shared_ex_r2l[i]});
            task3_unlab_pred_exs.push_back(task3_unlab_w1_pred * task3_word_ex);
            task3_unlab_arg_exs.push_back(task3_unlab_w1_arg * task3_word_ex);
            task3_lab_pred_exs.push_back(task3_lab_w1_pred * task3_word_ex);
            task3_lab_arg_exs.push_back(task3_lab_w1_arg * task3_word_ex);
        }

        /* task1 begin */
        vector<Expression> task1_exps(task1_parts->size());
        task1_scores.assign(task1_parts->size(), 0.0);
        task1_predicted_outputs.assign(task1_parts->size(), 0.0);
        SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
        vector<Expression> task1_unlab_phi_cache(slen * slen);
        vector<bool> task1_unlab_phi_flag(slen * slen, false);
        vector<Expression> task1_lab_phi_cache(slen * slen);
        vector<bool> task1_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task1_parts->size(); ++r) {
            if ((*task1_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task1_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task1_pred_ex = concatenate({task1_ex_l2r[idx_pred], task1_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task1_pred_MLP_in = tanh(task1_pred_w1 * task1_pred_ex + task1_pred_b1);
                Expression task1_pred_phi = tanh(task1_pred_w2 * task1_pred_MLP_in + task1_pred_b2);
                task1_exps[r] = task1_pred_w3 * task1_pred_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
            } else if ((*task1_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task1_unlab_MLP_in = tanh(
                        task1_unlab_pred_exs[idx_pred] + task1_unlab_arg_exs[idx_arg] + task1_unlab_b1);
                Expression task1_lab_MLP_in = tanh(
                        task1_lab_pred_exs[idx_pred] + task1_lab_arg_exs[idx_arg] + task1_lab_b1);
                Expression task1_unlab_phi = tanh(task1_unlab_w2 * task1_unlab_MLP_in + task1_unlab_b2);
                Expression task1_lab_phi = tanh(task1_lab_w2 * task1_lab_MLP_in + task1_lab_b2);
                task1_unlab_phi_cache[idx_pred * slen + idx_arg] = task1_unlab_phi;
                task1_lab_phi_cache[idx_pred * slen + idx_arg] = task1_lab_phi;
                task1_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task1_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task1_exps[r] = task1_unlab_w3 * task1_unlab_phi;
                task1_scores[r] = dynet::as_scalar(cg.incremental_forward(task1_exps[r]));
                Expression lab_MLP_out = task1_lab_w3 * task1_lab_phi + task1_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task1_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task1_parts->size());
                    CHECK_EQ((*task1_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task1_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task1_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task1_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task1 end */

        /* task2 begin */
        vector<Expression> task2_exps(task2_parts->size());
        task2_scores.assign(task2_parts->size(), 0.0);
        task2_predicted_outputs.assign(task2_parts->size(), 0.0);
        SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
        vector<Expression> task2_unlab_phi_cache(slen * slen);
        vector<bool> task2_unlab_phi_flag(slen * slen, false);
        vector<Expression> task2_lab_phi_cache(slen * slen);
        vector<bool> task2_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task2_parts->size(); ++r) {
            if ((*task2_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task2_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task2_pred_ex = concatenate({task2_ex_l2r[idx_pred], task2_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task2_pred_MLP_in = tanh(task2_pred_w1 * task2_pred_ex + task2_pred_b1);
                Expression task2_pred_phi = tanh(task2_pred_w2 * task2_pred_MLP_in + task2_pred_b2);
                task2_exps[r] = task2_pred_w3 * task2_pred_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
            } else if ((*task2_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task2_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task2_unlab_MLP_in = tanh(
                        task2_unlab_pred_exs[idx_pred] + task2_unlab_arg_exs[idx_arg] + task2_unlab_b1);
                Expression task2_lab_MLP_in = tanh(
                        task2_lab_pred_exs[idx_pred] + task2_lab_arg_exs[idx_arg] + task2_lab_b1);
                Expression task2_unlab_phi = tanh(task2_unlab_w2 * task2_unlab_MLP_in + task2_unlab_b2);
                Expression task2_lab_phi = tanh(task2_lab_w2 * task2_lab_MLP_in + task2_lab_b2);
                task2_unlab_phi_cache[idx_pred * slen + idx_arg] = task2_unlab_phi;
                task2_lab_phi_cache[idx_pred * slen + idx_arg] = task2_lab_phi;
                task2_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task2_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task2_exps[r] = task2_unlab_w3 * task2_unlab_phi;
                task2_scores[r] = dynet::as_scalar(cg.incremental_forward(task2_exps[r]));
                Expression lab_MLP_out = task2_lab_w3 * task2_lab_phi + task2_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task2_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task2_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task2_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task2 end */

        /* task3 begin */
        vector<Expression> task3_exps(task3_parts->size());
        task3_scores.assign(task3_parts->size(), 0.0);
        task3_predicted_outputs.assign(task3_parts->size(), 0.0);
        SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
        vector<Expression> task3_unlab_phi_cache(slen * slen);
        vector<bool> task3_unlab_phi_flag(slen * slen, false);
        vector<Expression> task3_lab_phi_cache(slen * slen);
        vector<bool> task3_lab_phi_flag(slen * slen, false);
        for (int r = 0; r < task3_parts->size(); ++r) {
            if ((*task3_parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*task3_parts)[r]);
                int idx_pred = predicate->predicate();
                Expression task3_pred_ex = concatenate({task3_ex_l2r[idx_pred], task3_ex_r2l[idx_pred],
                                                        shared_ex_l2r[idx_pred], shared_ex_r2l[idx_pred]});
                Expression task3_pred_MLP_in = tanh(task3_pred_w1 * task3_pred_ex + task3_pred_b1);
                Expression task3_pred_phi = tanh(task3_pred_w2 * task3_pred_MLP_in + task3_pred_b2);
                task3_exps[r] = task3_pred_w3 * task3_pred_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
            } else if ((*task3_parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task3_parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression task3_unlab_MLP_in = tanh(
                        task3_unlab_pred_exs[idx_pred] + task3_unlab_arg_exs[idx_arg] + task3_unlab_b1);
                Expression task3_lab_MLP_in = tanh(
                        task3_lab_pred_exs[idx_pred] + task3_lab_arg_exs[idx_arg] + task3_lab_b1);
                Expression task3_unlab_phi = tanh(task3_unlab_w2 * task3_unlab_MLP_in + task3_unlab_b2);
                Expression task3_lab_phi = tanh(task3_lab_w2 * task3_lab_MLP_in + task3_lab_b2);
                task3_unlab_phi_cache[idx_pred * slen + idx_arg] = task3_unlab_phi;
                task3_lab_phi_cache[idx_pred * slen + idx_arg] = task3_lab_phi;
                task3_unlab_phi_flag[idx_pred * slen + idx_arg] = true;
                task3_lab_phi_flag[idx_pred * slen + idx_arg] = true;
                task3_exps[r] = task3_unlab_w3 * task3_unlab_phi;
                task3_scores[r] = dynet::as_scalar(cg.incremental_forward(task3_exps[r]));
                Expression lab_MLP_out = task3_lab_w3 * task3_lab_phi + task3_lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_out));
                const vector<int> &index_labeled_parts =
                        task3_semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    int role = labeled_arc->role();
                    task3_exps[index_labeled_parts[k]] = pick(lab_MLP_out, role);
                    task3_scores[index_labeled_parts[k]] = label_scores[role];
                }
            }
        }
        /* task3 end */

        /* ctf begin */
        vector<Expression> ctf_exps(ctf_parts->size());
        ctf_scores.assign(ctf_parts->size(), 0.0);
        ctf_predicted_outputs.assign(ctf_parts->size(), 0.0);

        Expression task1task2_unlab_psi_cache = cmult(task1_unlab_V * transpose(task1_unlab_w3),
                                                      task2_unlab_V * transpose(task2_unlab_w3));
        Expression task1task3_unlab_psi_cache = cmult(task1_unlab_V * transpose(task1_unlab_w3),
                                                      task3_unlab_V * transpose(task3_unlab_w3));
        Expression task2task3_unlab_psi_cache = cmult(task2_unlab_V * transpose(task2_unlab_w3),
                                                      task3_unlab_V * transpose(task3_unlab_w3));
        Expression all_unlab_psi_cache = cmult(task1task2_unlab_psi_cache,
                                               task3_unlab_V * transpose(task3_unlab_w3));

        vector<Expression> task1task2_lab_phi_cache(slen * slen);
        vector<bool> task1task2_lab_phi_flag(slen * slen, false);
        vector<Expression> task1task3_lab_phi_cache(slen * slen);
        vector<bool> task1task3_lab_phi_flag(slen * slen, false);
        vector<Expression> task2task3_lab_phi_cache(slen * slen);
        vector<bool> task2task3_lab_phi_flag(slen * slen, false);
        vector<Expression> all_lab_phi_cache(slen * slen);
        vector<bool> all_lab_phi_flag(slen * slen, false);

        Expression task1_lab_psi_cache = task1_lab_w3 * transpose(task1_lab_V);
        Expression task2_lab_psi_cache = task2_lab_w3 * transpose(task2_lab_V);
        Expression task3_lab_psi_cache = task3_lab_w3 * transpose(task3_lab_V);

        vector<Expression> task1task2_lab_psi_cache(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE);
        vector<bool> task1task2_lab_psi_flag(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE, false);
        vector<Expression> task1task3_lab_psi_cache(TASK1_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> task1task3_lab_psi_flag(TASK1_LABEL_SIZE * TASK3_LABEL_SIZE, false);
        vector<Expression> task2task3_lab_psi_cache(TASK2_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> task2task3_lab_psi_flag(TASK2_LABEL_SIZE * TASK3_LABEL_SIZE, false);
        vector<Expression> all_lab_psi_cache(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE);
        vector<bool> all_lab_psi_flag(TASK1_LABEL_SIZE * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE, false);

        for (int r = 0; r < ctf_parts->size(); ++r) {
            if ((*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORM2NDORDER
                || (*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORM3RDORDER) {
                SemanticPartCrossForm2ndOrder *cross_form_arc = static_cast<SemanticPartCrossForm2ndOrder *>((*ctf_parts)[r]);
                int idx_pred = cross_form_arc->predicate();
                int idx_arg = cross_form_arc->argument();
                int idx = idx_pred * slen + idx_arg;
                int task1_role = cross_form_arc->role("task1");
                int task2_role = cross_form_arc->role("task2");
                int task3_role = cross_form_arc->role("task3");
                CHECK((task1_role < 0 || task1_unlab_phi_flag[idx])
                      && (task2_role < 0 || task2_unlab_phi_flag[idx])
                      && (task3_role < 0 || task3_unlab_phi_flag[idx]));

                if (task1_role > 0 && task2_role > 0 && task3_role < 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task2_unlab_U * task2_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task1task2_unlab_psi_cache)));
                } else if (task1_role > 0 && task2_role < 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task1task3_unlab_psi_cache)));
                } else if (task1_role < 0 && task2_role > 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task2_unlab_U * task2_unlab_phi_cache[idx],
                                               task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, task2task3_unlab_psi_cache)));
                } else if (task1_role > 0 && task2_role > 0 && task3_role > 0) {
                    Expression t_unlab = cmult(task1_unlab_U * task1_unlab_phi_cache[idx],
                                               task2_unlab_U * task2_unlab_phi_cache[idx]);
                    t_unlab = cmult(t_unlab, task3_unlab_U * task3_unlab_phi_cache[idx]);
                    ctf_exps[r] = sum_cols(transpose(cmult(t_unlab, all_unlab_psi_cache)));
                } else {
                    CHECK(false) << "Cross form arc error. Giving up.";
                }
                ctf_scores[r] = dynet::as_scalar(cg.incremental_forward(ctf_exps[r]));
            } else if ((*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORMLABELED2NDORDER
                       || (*ctf_parts)[r]->type() == SEMANTICPART_CROSSFORMLABELED3RDORDER) {
                SemanticPartCrossFormLabeled2ndOrder *cross_form_labeled_arc
                        = static_cast<SemanticPartCrossFormLabeled2ndOrder *>((*ctf_parts)[r]);
                int idx_pred = cross_form_labeled_arc->predicate();
                int idx_arg = cross_form_labeled_arc->argument();
                int idx = idx_pred * slen + idx_arg;
                int task1_role = cross_form_labeled_arc->role("task1");
                int task2_role = cross_form_labeled_arc->role("task2");
                int task3_role = cross_form_labeled_arc->role("task3");
                CHECK((task1_role < 0 || task1_lab_phi_flag[idx])
                      && (task2_role < 0 || task2_lab_phi_flag[idx])
                      && (task3_role < 0 || task3_lab_phi_flag[idx]));

                if (task1_role >= 0 && task2_role >= 0 && task3_role < 0) {
                    if (!task1task2_lab_phi_flag[idx]) {
                        task1task2_lab_phi_cache[idx] = cmult(task1_lab_U * task1_lab_phi_cache[idx],
                                                              task2_lab_U * task2_lab_phi_cache[idx]);
                        task1task2_lab_phi_flag[idx] = true;
                    }
                    if (!task1task2_lab_psi_flag[task1_role * TASK2_LABEL_SIZE + task2_role]) {
                        task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role]
                                = cmult(pick(task1_lab_psi_cache, task1_role),
                                        pick(task2_lab_psi_cache, task2_role));
                        task1task2_lab_psi_flag[task1_role * TASK2_LABEL_SIZE + task2_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task1task2_lab_phi_cache[idx],
                                  task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role])));
                } else if (task1_role >= 0 && task2_role < 0 && task3_role >= 0) {
                    if (!task1task3_lab_phi_flag[idx]) {
                        task1task3_lab_phi_cache[idx] = cmult(task1_lab_U * task1_lab_phi_cache[idx],
                                                              task3_lab_U * task3_lab_phi_cache[idx]);
                        task1task3_lab_phi_flag[idx] = true;
                    }
                    if (!task1task3_lab_psi_flag[task1_role * TASK3_LABEL_SIZE + task3_role]) {
                        task1task3_lab_psi_cache[task1_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(pick(task1_lab_psi_cache, task1_role),
                                        pick(task3_lab_psi_cache, task3_role));
                        task1task3_lab_psi_flag[task1_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task1task3_lab_phi_cache[idx],
                                  task1task3_lab_psi_cache[task1_role * TASK3_LABEL_SIZE + task3_role])));
                } else if (task1_role < 0 && task2_role >= 0 && task3_role >= 0) {
                    if (!task2task3_lab_phi_flag[idx]) {
                        task2task3_lab_phi_cache[idx] = cmult(task2_lab_U * task2_lab_phi_cache[idx],
                                                              task3_lab_U * task3_lab_phi_cache[idx]);
                        task2task3_lab_phi_flag[idx] = true;
                    }
                    if (!task2task3_lab_psi_flag[task2_role * TASK3_LABEL_SIZE + task3_role]) {
                        task2task3_lab_psi_cache[task2_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(pick(task2_lab_psi_cache, task2_role),
                                        pick(task3_lab_psi_cache, task3_role));
                        task2task3_lab_psi_flag[task2_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(task2task3_lab_phi_cache[idx],
                                  task2task3_lab_psi_cache[task2_role * TASK3_LABEL_SIZE + task3_role])));
                } else if (task1_role >= 0 && task2_role >= 0 && task3_role >= 0) {
                    if (!all_lab_phi_flag[idx]) {
                        all_lab_phi_cache[idx] = cmult(task1task2_lab_phi_cache[idx],
                                                       task3_lab_U * task3_lab_phi_cache[idx]);
                        all_lab_phi_flag[idx] = true;
                    }
                    if (!all_lab_psi_flag[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                          + task2_role * TASK3_LABEL_SIZE + task3_role]) {
                        all_lab_psi_cache[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                          + task2_role * TASK3_LABEL_SIZE + task3_role]
                                = cmult(task1task2_lab_psi_cache[task1_role * TASK2_LABEL_SIZE + task2_role],
                                        pick(task3_lab_psi_cache, task3_role));
                        all_lab_psi_flag[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                         + task2_role * TASK3_LABEL_SIZE + task3_role] = true;
                    }
                    ctf_exps[r] = sum_cols(transpose(
                            cmult(all_lab_phi_cache[idx],
                                  all_lab_psi_cache[task1_role * TASK2_LABEL_SIZE * TASK3_LABEL_SIZE
                                                    + task2_role * TASK3_LABEL_SIZE + task3_role])));
                } else {
                    CHECK(false) << "Cross form labeled arc error. Giving up.";
                }
                ctf_scores[r] = dynet::as_scalar(cg.incremental_forward(ctf_exps[r]));
            } else {
                CHECK(false) << "part type mistake: " << (*ctf_parts)[r]->type() << endl;
            }
        }
        /* ctf end */

        SemanticDecoder *semantic_decoder = static_cast <SemanticDecoder *> (decoder_);
        if (!is_train) {
            semantic_decoder->DecodeCrossForm(instance,
                                              task1_parts, task1_scores, &task1_predicted_outputs,
                                              task2_parts, task2_scores, &task2_predicted_outputs,
                                              task3_parts, task3_scores, &task3_predicted_outputs,
                                              ctf_parts, ctf_scores, &ctf_predicted_outputs);

            for (int r = 0; r < task1_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            for (int r = 0; r < task2_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < task3_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                    i_errs.push_back(i_err);
                }
            }

            for (int r = 0; r < ctf_parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(ctf_gold_outputs[r], ctf_predicted_outputs[r], 1e-6)) {
                    Expression i_err = (ctf_predicted_outputs[r] - ctf_gold_outputs[r]) * ctf_exps[r];
                    i_errs.push_back(i_err);
                }
            }
            Expression loss = Expression(&cg, cg.add_input(0.0));
            if (i_errs.size() > 0) {
                loss = loss + sum(i_errs);
            }
            return loss;
        }

        double s_loss = 0.0, s_cost = 0.0, cost = 0.0;
        semantic_decoder->DecodeCostAugmentedCrossForm(instance,
                                                       task1_parts, task1_scores, task1_gold_outputs,
                                                       &task1_predicted_outputs,
                                                       task2_parts, task2_scores, task2_gold_outputs,
                                                       &task2_predicted_outputs,
                                                       task3_parts, task3_scores, task3_gold_outputs,
                                                       &task3_predicted_outputs,
                                                       ctf_parts, ctf_scores, ctf_gold_outputs,
                                                       &ctf_predicted_outputs,
                                                       &s_cost, &s_loss);
        cost += s_cost;
        for (int r = 0; r < task1_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task1_predicted_outputs[r] - task1_gold_outputs[r]) * task1_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task2_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task2_predicted_outputs[r] - task2_gold_outputs[r]) * task2_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < task3_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) {
                Expression i_err = (task3_predicted_outputs[r] - task3_gold_outputs[r]) * task3_exps[r];
                i_errs.push_back(i_err);
            }
        }
        for (int r = 0; r < ctf_parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(ctf_gold_outputs[r], ctf_predicted_outputs[r], 1e-6)) {
                Expression i_err = (ctf_predicted_outputs[r] - ctf_gold_outputs[r]) * ctf_exps[r];
                i_errs.push_back(i_err);
            }
        }
        Expression loss = Expression(&cg, cg.add_input(cost));
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};