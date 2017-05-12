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

using namespace std;
using namespace dynet::expr;
const int UNK_ID = 0;
const unsigned VOCAB_SIZE = 35000;
const unsigned POS_SIZE = 49;

class biLSTM {
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

    virtual Expression BuildGraph(Instance *instance, Parts *parts, Decoder *decoder_, vector<double> &scores,
                                  const vector<double> &gold_outputs, vector<double> &predicted_outputs,
                                  const bool &use_word_dropout, const float &word_dropout_rate,
                                  unordered_map<int, int> *form_count,
                                  const bool &is_train, dynet::ComputationGraph &cg) {}

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
class biLSTMParser : public biLSTM {
private:
    // predicate part
    dynet::Parameter p_pred_w1;
    dynet::Parameter p_pred_b1;
    dynet::Parameter p_pred_w2;
    dynet::Parameter p_pred_b2;
    dynet::Parameter p_pred_w3;

    // unlabeled arc
    dynet::Parameter p_unlab_w1_pred;
    dynet::Parameter p_unlab_w1_arg;
    dynet::Parameter p_unlab_b1;
    dynet::Parameter p_unlab_w2;
    dynet::Parameter p_unlab_b2;
    dynet::Parameter p_unlab_w3;
    dynet::Parameter p_unlab_b3;

    // labeled arc
    dynet::Parameter p_lab_w1_pred;
    dynet::Parameter p_lab_w1_arg;
    dynet::Parameter p_lab_b1;
    dynet::Parameter p_lab_w2;
    dynet::Parameter p_lab_b2;
    dynet::Parameter p_lab_w3;
    dynet::Parameter p_lab_b3;

    Builder l2rbuilder;
    Builder r2lbuilder;

public:
    explicit biLSTMParser(dynet::Model *model) {
    }

    explicit biLSTMParser(SemanticOptions *semantic_option, const int &num_roles, dynet::Model *model) :
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
        LABEL_SIZE = num_roles;
    }

    void InitParams(dynet::Model *model) {
        // shared
        p_embed_pre_word = model->add_lookup_parameters(VOCAB_SIZE, {PRE_WORD_DIM});
        p_embed_word = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        p_embed_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // predicate
        p_pred_w1 = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        p_pred_b1 = model->add_parameters({MLP_DIM});
        p_pred_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        p_pred_b2 = model->add_parameters({MLP_DIM});
        p_pred_w3 = model->add_parameters({1, MLP_DIM});
        // unlabeled arc
        p_unlab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        p_unlab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        p_unlab_b1 = model->add_parameters({MLP_DIM});
        p_unlab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        p_unlab_b2 = model->add_parameters({MLP_DIM});
        p_unlab_w3 = model->add_parameters({1, MLP_DIM});
        // labeled arc
        p_lab_w1_pred = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        p_lab_w1_arg = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        p_lab_b1 = model->add_parameters({MLP_DIM});
        p_lab_w2 = model->add_parameters({MLP_DIM, MLP_DIM});
        p_lab_b2 = model->add_parameters({MLP_DIM});
        p_lab_w3 = model->add_parameters({LABEL_SIZE, MLP_DIM});
        p_lab_b3 = model->add_parameters({LABEL_SIZE});
    }


    Expression BuildGraph(Instance *instance, Parts *parts, Decoder *decoder_, vector<double> &scores,
                          const vector<double> &gold_outputs, vector<double> &predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count,
                          const bool &is_train, dynet::ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        const int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> pos = sentence->GetPosIds();
        l2rbuilder.new_graph(cg);
        l2rbuilder.start_new_sequence();
        r2lbuilder.new_graph(cg);
        r2lbuilder.start_new_sequence();

        Expression pred_w1 = parameter(cg, p_pred_w1);
        Expression pred_b1 = parameter(cg, p_pred_b1);
        Expression pred_w2 = parameter(cg, p_pred_w2);
        Expression pred_b2 = parameter(cg, p_pred_b2);
        Expression pred_w3 = parameter(cg, p_pred_w3);

        Expression unlab_w1_pred = parameter(cg, p_unlab_w1_pred);
        Expression unlab_w1_arg = parameter(cg, p_unlab_w1_arg);
        Expression unlab_b1 = parameter(cg, p_unlab_b1);
        Expression unlab_w2 = parameter(cg, p_unlab_w2);
        Expression unlab_b2 = parameter(cg, p_unlab_b2);
        Expression unlab_w3 = parameter(cg, p_unlab_w3);

        Expression lab_w1_pred = parameter(cg, p_lab_w1_pred);
        Expression lab_w1_arg = parameter(cg, p_lab_w1_arg);
        Expression lab_b1 = parameter(cg, p_lab_b1);
        Expression lab_w2 = parameter(cg, p_lab_w2);
        Expression lab_b2 = parameter(cg, p_lab_b2);
        Expression lab_w3 = parameter(cg, p_lab_w3);
        Expression lab_b3 = parameter(cg, p_lab_b3);

        vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen);
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

        vector<Expression> unlab_pred_exs, unlab_arg_exs;
        vector<Expression> lab_pred_exs, lab_arg_exs;
        for (int i = 0; i < slen; ++i) {
            Expression word_ex = concatenate({ex_l2r[i], ex_r2l[i]});
            unlab_pred_exs.push_back(unlab_w1_pred * word_ex);
            unlab_arg_exs.push_back(unlab_w1_arg * word_ex);
            lab_pred_exs.push_back(lab_w1_pred * word_ex);
            lab_arg_exs.push_back(lab_w1_arg * word_ex);
        }

        vector<Expression> exps(parts->size());
        scores.assign(parts->size(), 0.0);
        predicted_outputs.assign(parts->size(), 0.0);
        SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
        for (int r = 0; r < parts->size(); ++r) {
            if ((*parts)[r]->type() == SEMANTICPART_PREDICATE) {
                SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *>((*parts)[r]);
                int idx_pred = predicate->predicate();
                Expression pred_ex = concatenate({ex_l2r[idx_pred], ex_r2l[idx_pred]});
                Expression pred_MLP_in = tanh(pred_w1 * pred_ex + pred_b1);
                Expression pred_phi = tanh(pred_w2 * pred_MLP_in + pred_b2);
                exps[r] = pred_w3 * pred_phi;
                scores[r] = dynet::as_scalar(cg.incremental_forward(exps[r]));
            } else if ((*parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();
                Expression unlab_MLP_in = tanh(unlab_pred_exs[idx_pred] + unlab_arg_exs[idx_arg] + unlab_b1);
                Expression lab_MLP_in = tanh(lab_pred_exs[idx_pred] + lab_arg_exs[idx_arg] + lab_b1);
                Expression unlab_phi = tanh(unlab_w2 * unlab_MLP_in + unlab_b2);
                Expression lab_phi = tanh(lab_w2 * lab_MLP_in + lab_b2);
                exps[r] = unlab_w3 * unlab_phi;
                scores[r] = dynet::as_scalar(cg.incremental_forward(exps[r]));
                Expression lab_MLP_o = lab_w3 * lab_phi + lab_b3;
                vector<float> label_scores = dynet::as_vector(cg.incremental_forward(lab_MLP_o));
                const vector<int> &index_labeled_parts =
                        semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    CHECK_GE(index_labeled_parts[k], 0);
                    CHECK_LT(index_labeled_parts[k], parts->size());
                    CHECK_EQ((*parts)[index_labeled_parts[k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*parts)[index_labeled_parts[k]]);
                    CHECK(labeled_arc != NULL);
                    scores[index_labeled_parts[k]] = label_scores[labeled_arc->role()];
                    exps[index_labeled_parts[k]] = pick(lab_MLP_o, labeled_arc->role());
                }
            }
        }
        vector<Expression> i_errs;
        if (!is_train) {
            decoder_->Decode(instance, parts, scores, &predicted_outputs);
            for (int r = 0; r < parts->size(); ++r) {
                if (!NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6)) {
                    Expression i_err = (predicted_outputs[r] - gold_outputs[r]) * exps[r];
                    i_errs.push_back(i_err);
                }
            }
            Expression loss = Expression(&cg, cg.add_input(0.0));
            if (i_errs.size() > 0) {
                loss = loss + sum(i_errs);
            }
            return loss;
        }
        double s_loss = 0.0, s_cost = 0.0;
        decoder_->DecodeCostAugmented(instance, parts, scores, gold_outputs,
                                      &predicted_outputs, &s_cost, &s_loss);
        for (int r = 0; r < parts->size(); ++r) {
            if (!NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6)) {
                Expression i_err = (predicted_outputs[r] - gold_outputs[r]) * exps[r];
                i_errs.push_back(i_err);
            }
        }
        Expression loss = Expression(&cg, cg.add_input(s_cost));
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};