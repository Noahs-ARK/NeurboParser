//
// Created by hpeng on 6/16/17.
//

#ifndef NEURBOPARSER_PARSER_H
#define NEURBOPARSER_PARSER_H

#endif //NEURBOPARSER_PARSER_H

#include "BiLSTM.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "SemanticOptions.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticPipe.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tr1/unordered_map>

class Parser : public BiLSTM {

protected:
    // predicate part
    Parameter pred_w1_;
    Parameter pred_b1_;
    Parameter pred_w2_;
    Parameter pred_b2_;
    Parameter pred_w3_;
    Parameter pred_b3_;

    // unlabeled arc
    Parameter unlab_w1_pred_;
    Parameter unlab_w1_arg_;
    Parameter unlab_b1_;
    Parameter unlab_w2_;
    Parameter unlab_b2_;
    Parameter unlab_w3_;
    Parameter unlab_b3_;

    // labeled arc
    Parameter lab_w1_pred_;
    Parameter lab_w1_arg_;
    Parameter lab_b1_;
    Parameter lab_w2_;
    Parameter lab_b2_;
    Parameter lab_w3_;
    Parameter lab_b3_;

public:
    explicit Parser(ParameterCollection *model) {
    }

    explicit Parser(SemanticOptions *semantic_option, const int &num_roles, Decoder *decoder, ParameterCollection *model) :
            BiLSTM(semantic_option->num_lstm_layers(),
                   semantic_option->word_dim() + semantic_option->lemma_dim() + semantic_option->pos_dim(),
                   semantic_option->lstm_dim(), decoder, model) {
        WORD_DIM = semantic_option->word_dim();
        LEMMA_DIM = semantic_option->lemma_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->lstm_dim();
        MLP_DIM = semantic_option->mlp_dim();
        LABEL_SIZE = num_roles;
    }

    void InitParams(ParameterCollection *model) {
        // shared
        embed_word_ = model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        embed_lemma_ = model->add_lookup_parameters(VOCAB_SIZE, {LEMMA_DIM});
        embed_pos_ = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        // predicate
        pred_w1_ = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        pred_b1_ = model->add_parameters({MLP_DIM});
        pred_w2_ = model->add_parameters({MLP_DIM, MLP_DIM});
        pred_b2_ = model->add_parameters({MLP_DIM});
        pred_w3_ = model->add_parameters({1, MLP_DIM});
        pred_b3_ = model->add_parameters({1});
        // unlabeled arc
        unlab_w1_pred_ = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        unlab_w1_arg_ = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        unlab_b1_ = model->add_parameters({MLP_DIM});
        unlab_w2_ = model->add_parameters({MLP_DIM, MLP_DIM});
        unlab_b2_ = model->add_parameters({MLP_DIM});
        unlab_w3_ = model->add_parameters({1, MLP_DIM});
        unlab_b3_ = model->add_parameters({1});
        // labeled arc
        lab_w1_pred_ = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        lab_w1_arg_ = model->add_parameters({MLP_DIM, 2 * LSTM_DIM});
        lab_b1_ = model->add_parameters({MLP_DIM});
        lab_w2_ = model->add_parameters({MLP_DIM, MLP_DIM});
        lab_b2_ = model->add_parameters({MLP_DIM});
        lab_w3_ = model->add_parameters({LABEL_SIZE, MLP_DIM});
        lab_b3_ = model->add_parameters({LABEL_SIZE});
    }


    Expression BuildGraph(Instance *instance, Parts *parts, vector<double> &scores,
                          const vector<double> &gold_outputs, vector<double> &predicted_outputs,
                          const bool &use_word_dropout, const float &word_dropout_rate,
                          unordered_map<int, int> *form_count,
                          const bool &is_train, ComputationGraph &cg) {
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        const int slen = sentence->size();
        const vector<int> words = sentence->GetFormIds();
        const vector<int> lemmas = sentence->GetLemmaIds();
        const vector<int> pos = sentence->GetPosIds();
        l2rbuilder_.new_graph(cg);
        l2rbuilder_.start_new_sequence();
        r2lbuilder_.new_graph(cg);
        r2lbuilder_.start_new_sequence();

        Expression pred_w1 = parameter(cg, pred_w1_);
        Expression pred_b1 = parameter(cg, pred_b1_);
        Expression pred_w2 = parameter(cg, pred_w2_);
        Expression pred_b2 = parameter(cg, pred_b2_);
        Expression pred_w3 = parameter(cg, pred_w3_);
        Expression pred_b3 = parameter(cg, pred_b3_);

        Expression unlab_w1_pred = parameter(cg, unlab_w1_pred_);
        Expression unlab_w1_arg = parameter(cg, unlab_w1_arg_);
        Expression unlab_b1 = parameter(cg, unlab_b1_);
        Expression unlab_w2 = parameter(cg, unlab_w2_);
        Expression unlab_b2 = parameter(cg, unlab_b2_);
        Expression unlab_w3 = parameter(cg, unlab_w3_);
        Expression unlab_b3 = parameter(cg, unlab_b3_);

        Expression lab_w1_pred = parameter(cg, lab_w1_pred_);
        Expression lab_w1_arg = parameter(cg, lab_w1_arg_);
        Expression lab_b1 = parameter(cg, lab_b1_);
        Expression lab_w2 = parameter(cg, lab_w2_);
        Expression lab_b2 = parameter(cg, lab_b2_);
        Expression lab_w3 = parameter(cg, lab_w3_);
        Expression lab_b3 = parameter(cg, lab_b3_);

        vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen);
        for (int i = 0; i < slen; ++i) {
            int word_idx = words[i];
            int lemma_idx = lemmas[i];
            if (use_word_dropout && word_idx != UNK_ID) {
                int count = form_count->find(word_idx)->second;
                float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                if (rand_float < word_dropout_rate / (static_cast<float> (count) + word_dropout_rate)) {
                    word_idx = UNK_ID;
                    lemma_idx = UNK_ID;
                }
            }
            Expression x_pre_word = lookup(cg, embed_word_, word_idx);
            Expression x_lemma = lookup(cg, embed_lemma_, lemma_idx);
            Expression x_pos = lookup(cg, embed_pos_, pos[i]);
            ex_words[i] = concatenate({x_pre_word, x_lemma, x_pos});
            ex_l2r[i] = l2rbuilder_.add_input(ex_words[i]);
        }
        for (int i = 0; i < slen; ++i) {
            ex_r2l[slen - i - 1] = r2lbuilder_.add_input(ex_words[slen - i - 1]);
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
                exps[r] = pred_w3 * pred_phi + pred_b3;
                scores[r] = as_scalar(cg.incremental_forward(exps[r]));
            } else if ((*parts)[r]->type() == SEMANTICPART_ARC) {
                SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);
                int idx_pred = arc->predicate();
                int idx_arg = arc->argument();

                Expression unlab_MLP_in = tanh(unlab_pred_exs[idx_pred] + unlab_arg_exs[idx_arg] + unlab_b1);
                Expression lab_MLP_in = tanh(lab_pred_exs[idx_pred] + lab_arg_exs[idx_arg] + lab_b1);

                Expression unlab_phi = tanh(unlab_w2 * unlab_MLP_in + unlab_b2);
                Expression lab_phi = tanh(lab_w2 * lab_MLP_in + lab_b2);
                exps[r] = unlab_w3 * unlab_phi + unlab_b3;
                scores[r] = as_scalar(cg.incremental_forward(exps[r]));
                Expression lab_MLP_o = lab_w3 * lab_phi + lab_b3;
                vector<float> label_scores = as_vector(cg.incremental_forward(lab_MLP_o));
                const vector<int> &index_labeled_parts =
                        semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());
                for (int k = 0; k < index_labeled_parts.size(); ++k) {
                    SemanticPartLabeledArc *labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*parts)[index_labeled_parts[k]]);
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
            Expression loss = input(cg, 0.0);
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
        Expression loss = input(cg, s_cost);
        if (i_errs.size() > 0) {
            loss = loss + sum(i_errs);
        }
        return loss;
    }
};