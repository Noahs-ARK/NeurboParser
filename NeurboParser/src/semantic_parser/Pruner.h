//
// Created by hpeng on 7/5/17.
//

#ifndef NEURBOPARSER_PRUNER_H
#define NEURBOPARSER_PRUNER_H

#endif //NEURBOPARSER_PRUNER_H
//
// Created by hpeng on 11/3/16.
//

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tr1/unordered_map>
#include "SemanticOptions.h"
#include "SemanticInstanceNumeric.h"

class Pruner : public BiLSTM {
private:
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

public:
    explicit Pruner(ParameterCollection *model) {

    }

    explicit Pruner(SemanticOptions *semantic_option, int num_roles, Decoder *decoder, ParameterCollection *model) :
            BiLSTM(semantic_option->pruner_num_lstm_layers(),
                   semantic_option->word_dim() + semantic_option->lemma_dim() + semantic_option->pos_dim(),
                   semantic_option->pruner_lstm_dim(), decoder, model) {
        WORD_DIM = semantic_option->word_dim();
        LEMMA_DIM = semantic_option->lemma_dim();
        POS_DIM = semantic_option->pos_dim();
        LSTM_DIM = semantic_option->pruner_lstm_dim();
        MLP_DIM = semantic_option->pruner_mlp_dim();
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

        vector<Expression> unlab_pred_exs(slen), unlab_arg_exs(slen);
        for (int i = 0; i < slen; ++i) {
            Expression word_ex = concatenate({ex_l2r[i], ex_r2l[i]});
            unlab_pred_exs[i] = (unlab_w1_pred * word_ex);
            unlab_arg_exs[i] = (unlab_w1_arg * word_ex);
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
                Expression unlab_phi = tanh(unlab_w2 * unlab_MLP_in + unlab_b2);
                exps[r] = unlab_w3 * unlab_phi + unlab_b3;
                scores[r] = as_scalar(cg.incremental_forward(exps[r]));
            }
        }
        SemanticDecoder *semantic_decoder =
                static_cast<SemanticDecoder *> (decoder_);
        vector<Expression> i_errs;
        if (!is_train) {
            semantic_decoder->DecodePruner(instance, parts, scores, &predicted_outputs);
            return input(cg, 0.0);
        }

        Expression entropy = input(cg, 0.0);
        DecodeBasicMarginals(instance, parts, exps,
                             &predicted_outputs, entropy, cg);
        for (int r = 0; r < parts->size(); ++r) {
            if (gold_outputs[r] != predicted_outputs[r]) {
                Expression i_err = (predicted_outputs[r] - gold_outputs[r]) * exps[r];
                i_errs.push_back(i_err);
            }
        }
        if (i_errs.size() > 0) {
            entropy = entropy + sum(i_errs);
        }
        return entropy;
    }

    void DecodeBasicMarginals(Instance *instance, Parts *parts,
                              const vector<Expression> &scores,
                              vector<double> *predicted_output,
                              Expression &entropy, ComputationGraph &cg) {
        int sentence_length =
                static_cast<SemanticInstanceNumeric *>(instance)->size();
        SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
        int offset_predicate_parts, num_predicate_parts;
        int offset_arcs, num_arcs;
        semantic_parts->GetOffsetPredicate(&offset_predicate_parts,
                                           &num_predicate_parts);
        semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);

        vector<SemanticPartArc *> arcs(num_arcs);
        vector<Expression> scores_arcs(num_arcs);
        for (int r = 0; r < num_arcs; ++r) {
            arcs[r] = static_cast<SemanticPartArc *>((*parts)[offset_arcs + r]);
            scores_arcs[r] = scores[offset_arcs + r];
        }

        vector<vector<vector<int> > > arcs_by_predicate;
        arcs_by_predicate.resize(sentence_length);
        for (int r = 0; r < num_arcs; ++r) {
            int p = arcs[r]->predicate();
            int s = arcs[r]->sense();
            if (s >= arcs_by_predicate[p].size()) {
                arcs_by_predicate[p].resize(s + 1);
            }
            arcs_by_predicate[p][s].push_back(r);
        }

        vector<Expression> scores_predicates(num_predicate_parts);
        vector<vector<int> > index_predicates(sentence_length);
        for (int r = 0; r < num_predicate_parts; ++r) {
            scores_predicates[r] = scores[offset_predicate_parts + r];
            SemanticPartPredicate *predicate_part =
                    static_cast<SemanticPartPredicate *>((*parts)[offset_predicate_parts + r]);
            int p = predicate_part->predicate();
            int s = predicate_part->sense();
            if (s >= index_predicates[p].size()) {
                index_predicates[p].resize(s + 1, -1);
            }
            index_predicates[p][s] = r;
        }

        predicted_output->assign(parts->size(), 0.0);

        Expression log_partition_function = input(cg, 0.0);

        for (int p = 0; p < sentence_length; ++p) {
            Expression log_partition_all_senses = input(cg, 0.0);
            vector<Expression> log_partition_senses(arcs_by_predicate[p].size());
            vector<vector<Expression> > log_partition_arcs(arcs_by_predicate[p].size());

            for (int s = 0; s < arcs_by_predicate[p].size(); ++s) {
                int r = index_predicates[p][s];
                Expression score = scores_predicates[r];
                log_partition_arcs[s].assign(arcs_by_predicate[p][s].size(),
                                             input(cg, 0.0));
                for (int k = 0; k < arcs_by_predicate[p][s].size(); ++k) {
                    int r = arcs_by_predicate[p][s][k];
                    log_partition_arcs[s][k] = logsumexp({log_partition_arcs[s][k], scores_arcs[r]});
                    score = score + log_partition_arcs[s][k];
                }
                log_partition_senses[s] = score;
                log_partition_all_senses = logsumexp({log_partition_all_senses, log_partition_senses[s]});
            }

            if (arcs_by_predicate[p].size() > 0) {
                log_partition_function = log_partition_function + log_partition_all_senses;
            }

            for (int s = 0; s < arcs_by_predicate[p].size(); ++s) {
                int r = index_predicates[p][s];
                Expression predicate_marginal = exp(log_partition_senses[s] - log_partition_all_senses);

                (*predicted_output)[offset_predicate_parts + r]
                        = as_scalar(cg.incremental_forward(predicate_marginal));
                entropy = entropy - scores_predicates[r] * predicate_marginal;

                for (int k = 0; k < arcs_by_predicate[p][s].size(); ++k) {
                    int r = arcs_by_predicate[p][s][k];
                    Expression marginal = exp(scores_arcs[r] -  log_partition_arcs[s][k]);
                    marginal = marginal * predicate_marginal;
                    (*predicted_output)[offset_arcs + r]
                            = as_scalar(cg.incremental_forward(marginal));
                    entropy = entropy - scores_arcs[r] * marginal;
                }
            }
        }
        entropy = entropy + log_partition_function;
    }
};