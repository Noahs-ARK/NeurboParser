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

#include "SemanticDecoder.h"
#include "SemanticPipe.h"
#include "logval.h"
#include "FactorSemanticGraph.h"

// Define a matrix of doubles using Eigen.
typedef LogVal<double> LogValD;
namespace Eigen {
    typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
}

using namespace std;

DEFINE_double(srl_train_cost_false_positives, 1.0,
              "Cost for predicting false positives.");
DEFINE_double(srl_train_cost_false_negatives, 1.0,
              "Cost for predicting false negatives.");

void SemanticDecoder::BuildLabelIndices(Instance *instance, Parts *parts, AD3::FactorGraph *factor_graph,
                                        vector<int> *part_indices,
                                        vector<AD3::BinaryVariable *> *variables,
                                        vector<int> &pred_indices,
                                        vector<unordered_map<PKey, int>> &arg_indices_by_predicate,
                                        vector<set<int>> &roles_by_predicate,
                                        const vector<double> &scores) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    //int num_predicates = sentence->GetNumPredicates();

    // Get the offsets for the different parts.
    int offset_pred_parts, num_pred_parts;
    semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
    int offset_arg_parts, num_arg_parts;
    semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);

    if (factor_graph) {
        part_indices->clear();
    }

    pred_indices.assign(num_pred_parts, 0);
    roles_by_predicate.assign(num_pred_parts, set<int>());
    arg_indices_by_predicate.assign(num_pred_parts, unordered_map<PKey, int>());
    for (int i = 0;i < num_pred_parts; ++ i) {
        int r = i + offset_pred_parts;
        pred_indices[i] = r;
        if (factor_graph) {
            part_indices->push_back(r);
            AD3::BinaryVariable *variable = factor_graph->CreateBinaryVariable();
            variable->SetLogPotential(scores[r]);
            variables->push_back(variable);
        }
    }

    int s, e;
    // TODO: change this
    int offset_pred_variables = 0, offset_arg_variables = num_pred_parts;
    for (int i = 0;i < num_pred_parts; ++ i) {
        int pred_r = i + offset_pred_parts;
        vector<int> arg_part_indices
                = semantic_parts->GetArgumentsByPredicate(pred_r);

        for (auto arg_r: arg_part_indices) {
            int arg_idx = arg_r - offset_arg_parts + offset_arg_variables;
            SemanticPartArgument *arg_part =
                    static_cast<SemanticPartArgument *>((*parts)[arg_r]);
            arg_part->span(s, e);
            arg_indices_by_predicate[i].insert(
                    make_pair(PKey(s, e, arg_part->role()), arg_idx));
            if (factor_graph) {
                part_indices->push_back(arg_r);
                AD3::BinaryVariable *variable = factor_graph->CreateBinaryVariable();
                variable->SetLogPotential(scores[arg_r]);
                variables->push_back(variable);
            }
            roles_by_predicate[i].insert(arg_part->role());
        }
    }
}

void SemanticDecoder::DecodeLabel(int slen,
                                  const vector<int> &pred_indices,
                                  const vector<unordered_map<PKey, int>> &arg_indices_by_predicate,
                                  const vector<set<int>> &roles_by_predicate,
                                  const vector<double> &scores, vector<bool> &selected_parts, double *value) {
    int num_predicates = pred_indices.size();
    *value = 0.0;
    vector<bool> copied_selected_parts;
    double copied_value;
    for (int p = 0; p < num_predicates; ++p) {
        copied_selected_parts.assign(selected_parts.size(), false);
        copied_value = 0.0;
        DecodeLabelPredicate(slen, roles_by_predicate[p],
                             arg_indices_by_predicate[p], scores,
                             copied_selected_parts, &copied_value);
        copied_value += scores[pred_indices[p]];
        copied_selected_parts[pred_indices[p]] = true;
        if (p == 0 || copied_value > *value) {
            selected_parts = copied_selected_parts;
            *value = copied_value;
        }
    }
}

void SemanticDecoder::DecodeLabelPredicate(int slen, const set<int> &roles,
                                           const unordered_map<PKey, int> &indices,
                                           const vector<double> &scores,
                                           vector<bool> &selected_parts, double *value) {
    vector<tuple<int, int, int>> ijt; // ijt stores positions where we get the max (i.e. i, j and tag)
    vector<tuple<int, int, int>> it; // it stores positions where we get the max ending at j
    it.push_back(make_tuple(0, 0, 0)); // push one to make the index consistent, now it[j] means j rather than j+1
    it.push_back(make_tuple(0, 0, 0));

    int max_span_length = pipe_->GetSemanticOptions()->max_span_length() + 1;
//    slen -= 1;
    vector<double> alpha(slen + 1, 0.0), v_f;
    for (int j = 2; j <= slen; ++j) {
        v_f.clear();
        ijt.clear();
        const int i_start = max(1, j - max_span_length + 2);
        for (auto role: roles) {
            for (int i = i_start; i < j; ++i) {
                PKey key_tuple(i, j - 1, role);
                if (indices.find(key_tuple) == indices.end()) continue;
                v_f.push_back(scores[indices.at(key_tuple)] + alpha[i]);
                ijt.push_back(make_tuple(i, j, role));
            }
        }
        for (int i = i_start; i < j; ++i) {
            v_f.push_back(alpha[i]);
            ijt.push_back(make_tuple(i, j, IndexRoleNone));
        }
        unsigned max_ind = 0;
        auto max_val = v_f[0];
        for (auto ind = 1; ind < v_f.size(); ++ind) {
            auto val = v_f[ind];
            if (max_val < val) {
                max_val = val;
                max_ind = ind;
            }
        }
        alpha[j] = v_f[max_ind];
        it.push_back(ijt[max_ind]);
    }
    auto cur_j = slen;
    while (cur_j > 1) {
        int cur_i = std::get<0>(it[cur_j]);
        if (std::get<2>(it[cur_j]) == IndexRoleNone) {
            cur_j = cur_i;
            continue;
        }
        PKey key_tuple(cur_i, cur_j - 1, std::get<2>(it[cur_j]));
        int index = indices.at(key_tuple);
        selected_parts[index] = true;
        *value += scores[index];
        cur_j = cur_i;
    }
}

void SemanticDecoder::Decode(Instance *instance, Parts *parts,
                             const vector<double> &scores,
                             vector<double> *predicted_outputs) {
    DecodeFactorGraph(instance, parts, scores, true,
                      predicted_outputs);

    double threshold = 0.5;
    if (!pipe_->GetSemanticOptions()->train()) {
        SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
        int slen = static_cast<SemanticInstanceNumeric *> (instance)->size() - 1;
        int offset_pred_parts, num_pred_parts;
        semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
        int offset_arg_parts, num_arg_parts;
        semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);

        vector<double> copied_scores(parts->size(), 0.0);

        for (int i = 0;i < num_pred_parts; ++ i) {
            copied_scores[i + offset_pred_parts]
                    = (*predicted_outputs)[i + offset_pred_parts];
        }
        for (int i = 0; i < num_arg_parts; ++i) {
            copied_scores[i + offset_arg_parts]
                    = (*predicted_outputs)[i + offset_arg_parts] - threshold;
        }

        vector<int> pred_indices;
        vector<unordered_map<PKey, int>> arg_indices_by_predicate;
        vector<set<int>> roles_by_predicate;
        BuildLabelIndices(instance, parts, NULL, NULL, NULL,
                          pred_indices, arg_indices_by_predicate,
                          roles_by_predicate, copied_scores);

        double value = 0.0;
        vector<bool> selected_parts(parts->size(), false);

        DecodeLabel(slen, pred_indices, arg_indices_by_predicate,
                    roles_by_predicate, copied_scores,  selected_parts, &value);

        predicted_outputs->assign(parts->size(), 0.0);
        for (int i = 0;i < parts->size(); ++ i) {
            if (selected_parts[i]) (*predicted_outputs)[i] = 1.0;
        }
    }
}

void SemanticDecoder::DecodeCostAugmented(Instance *instance, Parts *parts,
                                          const vector<double> &scores,
                                          const vector<double> &gold_output,
                                          vector<double> *predicted_output,
                                          double *cost,
                                          double *loss) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int offset_arg_parts, num_arg_parts;
    semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);

//	LogValD a1 = LogValD::Zero();
//	LogValD b1 = LogValD::One();
//	LOG(INFO) << a1.as_float()<< " "<<a1.logabs()<<endl;
//	LOG(INFO) << b1.as_float()<<" " << b1.logabs() << endl;
//	LOG(INFO) << a1.signbit() <<" " << b1.signbit();
//	LogValD c = a1 + b1;
//	LOG(INFO) << c.as_float();

    double a = FLAGS_srl_train_cost_false_positives;
    double b = FLAGS_srl_train_cost_false_negatives;
    double q = 0.0;
//    a = 0.5; b = 2.0;
    vector<double> p(num_arg_parts, 0.0);
    vector<double> scores_cost = scores;

    for (int r = 0; r < num_arg_parts; ++r) {
        p[r] = a - (a + b) * gold_output[offset_arg_parts + r];
        scores_cost[offset_arg_parts + r] += p[r];
        q += b * gold_output[offset_arg_parts + r];
    }

    Decode(instance, parts, scores_cost, predicted_output);

    *cost = q;
    for (int r = 0; r < num_arg_parts; ++r) {
        *cost += p[r] * (*predicted_output)[offset_arg_parts + r];
    }

    *loss = *cost;
    for (int r = 0; r < num_arg_parts; ++r) {
        *loss += scores[offset_arg_parts + r] * ((*predicted_output)[offset_arg_parts + r]
                                                 - gold_output[offset_arg_parts + r]);
    }
}


// Decode building a factor graph and calling the AD3 algorithm.
void SemanticDecoder::DecodeFactorGraph(Instance *instance, Parts *parts,
                                        const vector<double> &scores,
                                        bool relax,
                                        vector<double> *predicted_outputs) {
    // all the parts here should be arguments. and they are from the same frame
    // TODO: add pairwise requirements/exclusiveness constraints

    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    int slen = sentence->size() - 1;
//    int num_predicates = sentence->GetNumPredicates();
    CHECK(relax);
    SemanticDictionary *semantic_dictionary = pipe_->GetSemanticDictionary();
    // Get the offsets for the different parts.
    int offset_pred_parts, num_pred_parts;
    semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
    int offset_arg_parts, num_arg_parts;
    semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);
    

    vector<AD3::BinaryVariable *> variables;
    vector<int> part_indices;
    vector<int> additional_part_indices;
    vector<int> factor_part_indices_;
    AD3::FactorGraph *factor_graph = new AD3::FactorGraph;
    int verbosity = 1; //1;
    if (VLOG_IS_ON(2)) {
        verbosity = 2;
    }
    factor_graph->SetVerbosity(verbosity);

    int offset_pred_variables = 0, offset_arg_variables = num_pred_parts;

    vector<int> pred_indices;
    vector<unordered_map<PKey, int>> arg_indices_by_predicate;
    vector<set<int>> roles_by_predicate;
    BuildLabelIndices(instance, parts, factor_graph, &part_indices,
                      &variables, pred_indices, arg_indices_by_predicate, roles_by_predicate,
                      scores);

    CHECK_EQ(variables.size(), num_arg_parts + num_pred_parts);

    AD3::FactorSemanticGraph *factor = new AD3::FactorSemanticGraph;
    factor->Initialize(slen, num_pred_parts, num_arg_parts,
                       pred_indices,
                       arg_indices_by_predicate, roles_by_predicate, this, false);
    factor_graph->DeclareFactor(factor, variables, true);
    factor_part_indices_.push_back(-1);

    CHECK_EQ(variables.size(), part_indices.size());
    CHECK_EQ(factor_graph->GetNumFactors(), factor_part_indices_.size());

    bool solved = false;
    vector<double> posteriors;
    vector<double> additional_posteriors;
    double value_ref;
    double *value = &value_ref;

    factor_graph->SetMaxIterationsAD3(500);
    factor_graph->SetEtaAD3(0.05);
    factor_graph->AdaptEtaAD3(true);
    factor_graph->SetResidualThresholdAD3(1e-3);

    // Run AD3.
    timeval start, end;
    gettimeofday(&start, NULL);
    if (!solved) {
        factor_graph->SolveLPMAPWithAD3(&posteriors, &additional_posteriors, value);
    }
    gettimeofday(&end, NULL);
    double elapsed_time = diff_ms(end, start);
    VLOG(2) << "Elapsed time (AD3) = " << elapsed_time
            << " (" << slen << ") ";

    delete factor_graph;
    *value = 0.0;
    CHECK_EQ(posteriors.size(), variables.size());
    predicted_outputs->assign(parts->size(), 0.0);

    for (int i = 0;i < num_pred_parts; ++ i) {
        int r = part_indices[i + offset_pred_variables];
        CHECK_EQ(r, offset_pred_parts + i);
        (*predicted_outputs)[r] = posteriors[i + offset_pred_variables];
        *value += (*predicted_outputs)[r] * scores[r];
    }

    for (int i = 0; i < num_arg_parts; ++i) {
        int r = part_indices[i + offset_arg_variables];
        CHECK_EQ(r, offset_arg_parts + i);
        (*predicted_outputs)[r] = posteriors[i + offset_arg_variables];
        *value += (*predicted_outputs)[r] * scores[r];
    }
    VLOG(2) << "Solution value (AD3) = " << *value;
}