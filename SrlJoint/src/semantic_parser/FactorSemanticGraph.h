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

#ifndef FACTOR_SEMANTIC_GRAPH_H
#define FACTOR_SEMANTIC_GRAPH_H

#include "SemanticDecoder.h"
#include "ad3/GenericFactor.h"

namespace AD3 {
    class FactorSemanticGraph : public GenericFactor {
    public:
        FactorSemanticGraph() {}

        virtual ~FactorSemanticGraph() {
            ClearActiveSet();
        }

        // Print as a string.
        void Print(ostream &stream) {
            stream << "SEMANTIC_GRAPH";
            Factor::Print(stream);
            stream << " " << length_;
            stream << " " << num_pred_parts_;
            stream << endl;
        }

        // Compute the score of a given assignment.
        // Note: additional_log_potentials is empty and is ignored.
        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {
            vector<bool> *selected_parts = static_cast<vector<bool> *>(configuration);
            vector<double> scores(variable_log_potentials);
            decoder_->DecodeLabel(length_, pred_indices_, arg_indices_by_predicate_,
                                  roles_by_predicate_, scores, *selected_parts, value);
        }

        // Compute the score of a given assignment.
        // Note: additional_log_potentials is empty and is ignored.
        void Evaluate(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      const Configuration configuration,
                      double *value) {
            const vector<bool> *selected_parts =
                    static_cast<vector<bool> *>(configuration);
            *value = 0.0;
            for (int r = 0; r < selected_parts->size(); ++r) {
                int index = r;
                if ((*selected_parts)[r]) *value += variable_log_potentials[index];
            }
        }

        // Given a configuration with a probability (weight),
        // increment the vectors of variable and additional posteriors.
        // Note: additional_log_potentials is empty and is ignored.
        void UpdateMarginalsFromConfiguration(
                const Configuration &configuration,
                double weight,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors) {
            const vector<bool> *selected_parts =
                    static_cast<vector<bool> *>(configuration);
            for (int r = 0; r < selected_parts->size(); ++r) {
                int index = r;
                if ((*selected_parts)[r]) (*variable_posteriors)[index] += weight;
            }
        }

        // Count how many common values two configurations have.
        int CountCommonValues(const Configuration &configuration1,
                              const Configuration &configuration2) {
            const vector<bool> *selected_parts1 =
                    static_cast<vector<bool> *>(configuration1);
            const vector<bool> *selected_parts2 =
                    static_cast<vector<bool> *>(configuration2);
            CHECK_EQ(selected_parts1->size(), selected_parts2->size());
            int count = 0;
            for (int r = 0; r < selected_parts1->size(); ++r) {
                if ((*selected_parts1)[r] && (*selected_parts2)[r]) {
                    ++count;
                }
            }
            return count;
        }

        // Check if two configurations are the same.
        bool SameConfiguration(
                const Configuration &configuration1,
                const Configuration &configuration2) {
            const vector<bool> *selected_parts1 =
                    static_cast<vector<bool> *>(configuration1);
            const vector<bool> *selected_parts2 =
                    static_cast<vector<bool> *>(configuration2);
            CHECK_EQ(selected_parts1->size(), selected_parts2->size());
            for (int r = 0; r < selected_parts1->size(); ++r) {
                if ((*selected_parts1)[r] != (*selected_parts2)[r]) {
                    return false;
                }
            }
            return true;
        }

        // Delete configuration.
        void DeleteConfiguration(
                Configuration configuration) {
            vector<bool> *selected_parts = static_cast<vector<bool> *>(configuration);
            delete selected_parts;
        }

        // Create configuration.
        Configuration CreateConfiguration() {
            vector<bool> *selected_parts
                    = new vector<bool>(num_pred_parts_ + num_arg_parts_);
            return static_cast<Configuration>(selected_parts);
        }

    public:
        void Initialize(int length, int num_pred_parts, int num_arg_parts,
                        const vector<int> &pred_indices,
                        const vector<unordered_map<PKey, int>> &arg_indices_by_predicate,
                        const vector<set<int>> &roles_by_predicate,
                        SemanticDecoder *decoder,
                        bool own_parts = false) {
            length_ = length;
            num_pred_parts_ = num_pred_parts;
            num_arg_parts_ = num_arg_parts;
            pred_indices_ = pred_indices;
            arg_indices_by_predicate_ = arg_indices_by_predicate;
            roles_by_predicate_ = roles_by_predicate;
            decoder_ = decoder;
            own_parts_ = own_parts;
        }

    private:
        bool own_parts_;
        int length_; // Sentence length (including root symbol).
        int num_pred_parts_;
        int num_arg_parts_;
        vector<int> pred_indices_;
        vector<unordered_map<PKey, int>> arg_indices_by_predicate_;
        vector<set<int>> roles_by_predicate_;
        SemanticDecoder *decoder_;
    };
} // namespace AD3

#endif // FACTOR_SEMANTIC_GRAPH_H
