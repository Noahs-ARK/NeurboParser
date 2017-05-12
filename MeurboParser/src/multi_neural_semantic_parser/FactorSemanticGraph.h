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
            if (own_parts_) {
                for (int r = 0; r < arcs_.size(); ++r) {
                    delete arcs_[r];
                }
                for (int r = 0; r < predicate_parts_.size(); ++r) {
                    delete predicate_parts_[r];
                }
            }
            ClearActiveSet();
        }

        // Print as a string.
        void Print(ostream &stream) {
            stream << "SEMANTIC_GRAPH";
            Factor::Print(stream);
            stream << " " << length_;
            stream << " " << predicate_parts_.size();
            stream << " " << arcs_.size();
            for (int k = 0; k < predicate_parts_.size(); ++k) {
                int p = predicate_parts_[k]->predicate();
                int s = predicate_parts_[k]->sense();
                stream << " " << p << " " << s;
            }
            for (int k = 0; k < arcs_.size(); ++k) {
                int p = arcs_[k]->predicate();
                int a = arcs_[k]->argument();
                int s = arcs_[k]->sense();
                stream << " " << p << " " << a << " " << s;
            }
            stream << endl;
        }

        // Compute the score of a given assignment.
        // Note: additional_log_potentials is empty and is ignored.
        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {
            vector<bool> *selected_parts = static_cast<vector<bool> *>(configuration);
            int num_predicate_parts = predicate_parts_.size();
            int num_arcs = arcs_.size();
            CHECK_EQ(num_predicate_parts + num_arcs, selected_parts->size());
            vector<bool> selected_predicates;
            vector<bool> selected_arcs;
            vector<double> predicate_scores(variable_log_potentials.begin(),
                                            variable_log_potentials.begin() +
                                            num_predicate_parts);
            vector<double> arc_scores(variable_log_potentials.begin() +
                                      num_predicate_parts,
                                      variable_log_potentials.end());

            decoder_->DecodeSemanticGraph(length_, predicate_parts_, arcs_,
                                          index_predicates_, arcs_by_predicate_,
                                          predicate_scores, arc_scores,
                                          &selected_predicates, &selected_arcs, value);

            for (int r = 0; r < num_predicate_parts; ++r) {
                (*selected_parts)[r] = selected_predicates[r];
            }
            for (int r = 0; r < num_arcs; ++r) {
                (*selected_parts)[num_predicate_parts + r] = selected_arcs[r];
            }
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
            int num_predicate_parts = predicate_parts_.size();
            int num_arcs = arcs_.size();
            vector<bool> *selected_parts = new vector<bool>(num_predicate_parts +
                                                            num_arcs);
            return static_cast<Configuration>(selected_parts);
        }

    public:
        void Initialize(int length,
                        const vector<SemanticPartPredicate *> &predicate_parts,
                        const vector<SemanticPartArc *> &arcs,
                        SemanticDecoder *decoder,
                        bool own_parts = false) {
            own_parts_ = own_parts;
            length_ = length;
            predicate_parts_ = predicate_parts;
            arcs_ = arcs;
            decoder_ = decoder;

            decoder_->BuildBasicIndices(length_, predicate_parts_, arcs_,
                                        &index_predicates_, &arcs_by_predicate_);
        }

    private:
        bool own_parts_;
        int length_; // Sentence length (including root symbol).
        vector<vector<int> > index_predicates_;
        vector<vector<vector<int> > > arcs_by_predicate_;
        vector<SemanticPartPredicate *> predicate_parts_;
        vector<SemanticPartArc *> arcs_;
        SemanticDecoder *decoder_;
    };

    class FactorCrossFormSemanticGraph : public GenericFactor {
    public:
        FactorCrossFormSemanticGraph() {}

        virtual ~FactorCrossFormSemanticGraph() {
            if (own_parts_) {
                for (int r = 0; r < task1_arcs_.size(); ++r) {
                    delete task1_arcs_[r];
                }
                for (int r = 0; r < task1_predicate_parts_.size(); ++r) {
                    delete task1_predicate_parts_[r];
                }

                for (int r = 0; r < task2_arcs_.size(); ++r) {
                    delete task2_arcs_[r];
                }
                for (int r = 0; r < task2_predicate_parts_.size(); ++r) {
                    delete task2_predicate_parts_[r];
                }

                for (int r = 0; r < task3_arcs_.size(); ++r) {
                    delete task3_arcs_[r];
                }
                for (int r = 0; r < task3_predicate_parts_.size(); ++r) {
                    delete task3_predicate_parts_[r];
                }
            }
            ClearActiveSet();
        }

        // Print as a string.
        void Print(ostream &stream) {
            stream << "SEMANTIC_GRAPH: DM";
            Factor::Print(stream);
            stream << " " << length_;
            stream << " " << task1_predicate_parts_.size();
            stream << " " << task1_arcs_.size();
            for (int k = 0; k < task1_predicate_parts_.size(); ++k) {
                int p = task1_predicate_parts_[k]->predicate();
                int s = task1_predicate_parts_[k]->sense();
                stream << " " << p << " " << s;
            }
            for (int k = 0; k < task1_arcs_.size(); ++k) {
                int p = task1_arcs_[k]->predicate();
                int a = task1_arcs_[k]->argument();
                int s = task1_arcs_[k]->sense();
                stream << " " << p << " " << a << " " << s;
            }
            stream << endl;

            stream << "SEMANTIC_GRAPH: PAS";
            Factor::Print(stream);
            stream << " " << length_;
            stream << " " << task2_predicate_parts_.size();
            stream << " " << task2_arcs_.size();
            for (int k = 0; k < task2_predicate_parts_.size(); ++k) {
                int p = task2_predicate_parts_[k]->predicate();
                int s = task2_predicate_parts_[k]->sense();
                stream << " " << p << " " << s;
            }
            for (int k = 0; k < task2_arcs_.size(); ++k) {
                int p = task2_arcs_[k]->predicate();
                int a = task2_arcs_[k]->argument();
                int s = task2_arcs_[k]->sense();
                stream << " " << p << " " << a << " " << s;
            }
            stream << endl;

            stream << "SEMANTIC_GRAPH: PSD";
            Factor::Print(stream);
            stream << " " << length_;
            stream << " " << task3_predicate_parts_.size();
            stream << " " << task3_arcs_.size();
            for (int k = 0; k < task3_predicate_parts_.size(); ++k) {
                int p = task3_predicate_parts_[k]->predicate();
                int s = task3_predicate_parts_[k]->sense();
                stream << " " << p << " " << s;
            }
            for (int k = 0; k < task3_arcs_.size(); ++k) {
                int p = task3_arcs_[k]->predicate();
                int a = task3_arcs_[k]->argument();
                int s = task3_arcs_[k]->sense();
                stream << " " << p << " " << a << " " << s;
            }
            stream << endl;
        }

        // Compute the score of a given assignment.
        // Note: additional_log_potentials is empty and is ignored.
        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {
            vector<bool> *selected_parts = static_cast<vector<bool> *>(configuration);
            int task1_num_predicate_parts = task1_predicate_parts_.size();
            int task1_num_arcs = task1_arcs_.size();
            int task2_num_predicate_parts = task2_predicate_parts_.size();
            int task2_num_arcs = task2_arcs_.size();
            int task3_num_predicate_parts = task3_predicate_parts_.size();
            int task3_num_arcs = task3_arcs_.size();

            CHECK_EQ(task1_num_predicate_parts + task1_num_arcs
                     + task2_num_predicate_parts + task2_num_arcs
                     + task3_num_predicate_parts + task3_num_arcs, selected_parts->size());

            double task1_value = 0.0;
            vector<bool> task1_selected_predicates;
            vector<bool> task1_selected_arcs;
            vector<double> task1_predicate_scores(variable_log_potentials.begin() + task1_offset_,
                                            variable_log_potentials.begin()  + task1_offset_ + task1_num_predicate_parts);

            vector<double> task1_arc_scores(variable_log_potentials.begin() +  task1_offset_ + task1_num_predicate_parts,
                                      variable_log_potentials.begin() +  task1_offset_ + task1_num_predicate_parts + task1_num_arcs);

            decoder_->DecodeSemanticGraph(length_, task1_predicate_parts_, task1_arcs_,
                                          task1_index_predicates_, task1_arcs_by_predicate_,
                                          task1_predicate_scores, task1_arc_scores,
                                          &task1_selected_predicates, &task1_selected_arcs, &task1_value);
            for (int r = 0; r < task1_num_predicate_parts; ++r) {
                (*selected_parts)[task1_offset_ + r] = task1_selected_predicates[r];
            }
            for (int r = 0; r < task1_num_arcs; ++r) {
                (*selected_parts)[task1_offset_ + task1_num_predicate_parts + r] = task1_selected_arcs[r];
            }

            double task2_value = 0.0;
            vector<bool> task2_selected_predicates;
            vector<bool> task2_selected_arcs;
            vector<double> task2_predicate_scores(variable_log_potentials.begin() + task2_offset_,
                                                variable_log_potentials.begin()  + task2_offset_ + task2_num_predicate_parts);

            vector<double> task2_arc_scores(variable_log_potentials.begin() +  task2_offset_ + task2_num_predicate_parts,
                                          variable_log_potentials.begin() +  task2_offset_ + task2_num_predicate_parts + task2_num_arcs);

            decoder_->DecodeSemanticGraph(length_, task2_predicate_parts_, task2_arcs_,
                                          task2_index_predicates_, task2_arcs_by_predicate_,
                                          task2_predicate_scores, task2_arc_scores,
                                          &task2_selected_predicates, &task2_selected_arcs, &task2_value);
            for (int r = 0; r < task2_num_predicate_parts; ++r) {
                (*selected_parts)[task2_offset_ + r] = task2_selected_predicates[r];
            }
            for (int r = 0; r < task2_num_arcs; ++r) {
                (*selected_parts)[task2_offset_ + task2_num_predicate_parts + r] = task2_selected_arcs[r];
            }

            double task3_value = 0.0;
            vector<bool> task3_selected_predicates;
            vector<bool> task3_selected_arcs;
            vector<double> task3_predicate_scores(variable_log_potentials.begin() + task3_offset_,
                                                variable_log_potentials.begin()  + task3_offset_ + task3_num_predicate_parts);

            vector<double> task3_arc_scores(variable_log_potentials.begin() +  task3_offset_ + task3_num_predicate_parts,
                                          variable_log_potentials.begin() +  task3_offset_ + task3_num_predicate_parts + task3_num_arcs);

            decoder_->DecodeSemanticGraph(length_, task3_predicate_parts_, task3_arcs_,
                                          task3_index_predicates_, task3_arcs_by_predicate_,
                                          task3_predicate_scores, task3_arc_scores,
                                          &task3_selected_predicates, &task3_selected_arcs, &task3_value);
            for (int r = 0; r < task3_num_predicate_parts; ++r) {
                (*selected_parts)[task3_offset_ + r] = task3_selected_predicates[r];
            }
            for (int r = 0; r < task3_num_arcs; ++r) {
                (*selected_parts)[task3_offset_ + task3_num_predicate_parts + r] = task3_selected_arcs[r];
            }

            *value = task1_value + task2_value + task3_value;
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
            int task1_num_predicate_parts = task1_predicate_parts_.size();
            int task1_num_arcs = task1_arcs_.size();
            int task2_num_predicate_parts = task2_predicate_parts_.size();
            int task2_num_arcs = task2_arcs_.size();
            int task3_num_predicate_parts = task3_predicate_parts_.size();
            int task3_num_arcs = task3_arcs_.size();

            vector<bool> *selected_parts = new vector<bool>(task1_num_predicate_parts + task1_num_arcs
                                                            + task2_num_predicate_parts + task2_num_arcs
                                                            + task3_num_predicate_parts + task3_num_arcs);
            return static_cast<Configuration>(selected_parts);
        }

    public:
        void Initialize(int length,
                        const int &task1_offset, const vector<SemanticPartPredicate *> &task1_predicate_parts,
                        const vector<SemanticPartArc *> &task1_arcs,
                        const int &task2_offset, const vector<SemanticPartPredicate *> &task2_predicate_parts,
                        const vector<SemanticPartArc *> &task2_arcs,
                        const int &task3_offset, const vector<SemanticPartPredicate *> &task3_predicate_parts,
                        const vector<SemanticPartArc *> &task3_arcs,
                        SemanticDecoder *decoder,
                        bool own_parts = false) {
            own_parts_ = own_parts;
            length_ = length;

            task1_offset_ = task1_offset;
            task1_predicate_parts_ = task1_predicate_parts;
            task1_arcs_ = task1_arcs;
            decoder_->BuildBasicIndices(length_, task1_predicate_parts_, task1_arcs_,
                                        &task1_index_predicates_, &task1_arcs_by_predicate_);

            task2_offset_ = task2_offset;
            task2_predicate_parts_ = task2_predicate_parts;
            task2_arcs_ = task2_arcs;
            decoder_->BuildBasicIndices(length_, task2_predicate_parts_, task2_arcs_,
                                        &task2_index_predicates_, &task2_arcs_by_predicate_);

            task3_offset_ = task3_offset;
            task3_predicate_parts_ = task3_predicate_parts;
            task3_arcs_ = task3_arcs;
            decoder_->BuildBasicIndices(length_, task3_predicate_parts_, task3_arcs_,
                                        &task3_index_predicates_, &task3_arcs_by_predicate_);

            decoder_ = decoder;
        }

    private:
        bool own_parts_;
        int length_; // Sentence length (including root symbol).
        int task1_offset_;
        vector<vector<int> > task1_index_predicates_;
        vector<vector<vector<int> > > task1_arcs_by_predicate_;
        vector<SemanticPartPredicate *> task1_predicate_parts_;
        vector<SemanticPartArc *> task1_arcs_;

        int task2_offset_;
        vector<vector<int> > task2_index_predicates_;
        vector<vector<vector<int> > > task2_arcs_by_predicate_;
        vector<SemanticPartPredicate *> task2_predicate_parts_;
        vector<SemanticPartArc *> task2_arcs_;

        int task3_offset_;
        vector<vector<int> > task3_index_predicates_;
        vector<vector<vector<int> > > task3_arcs_by_predicate_;
        vector<SemanticPartPredicate *> task3_predicate_parts_;
        vector<SemanticPartArc *> task3_arcs_;
        SemanticDecoder *decoder_;
    };
} // namespace AD3

#endif // FACTOR_SEMANTIC_GRAPH_H
