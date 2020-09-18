/// Copyright (c) 2012-2015 Andre Martins
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

#ifndef SEMANTICDECODER_H_
#define SEMANTICDECODER_H_

#include "Decoder.h"
#include "SemanticPart.h"
#include "ad3/FactorGraph.h"

class SemanticPipe;

class PKey {
public:
    PKey(int x1, int x2, int x3)
            : t(x1, x2, x3) {}

    tuple<int, int, int> t;

    bool operator==(const PKey &a) const {
        return std::get<0>(a.t) == std::get<0>(t) &&
               std::get<1>(a.t) == std::get<1>(t) &&
               std::get<2>(a.t) == std::get<2>(t);
    }
};


namespace std {
    template<>
    struct hash<PKey> {
        std::size_t operator()(const PKey &e) const {
            return (std::get<0>(e.t)) | (std::get<1>(e.t) << 16) | (std::get<2>(e.t) << 24);
        }
    };
}

class SemanticDecoder : public Decoder {
public:
    SemanticDecoder() {};

    SemanticDecoder(SemanticPipe *pipe) : pipe_(pipe) {};

    virtual ~SemanticDecoder() {};

    void Decode(Instance *instance, Parts *parts,
                const vector<double> &scores,
                vector<double> *predicted_output);

    void DecodeCostAugmented(Instance *instance, Parts *parts,
                             const vector<double> &scores,
                             const vector<double> &gold_output,
                             vector<double> *predicted_output,
                             double *cost,
                             double *loss);

    void DecodeFactorGraph(Instance *instance, Parts *parts,
                           const vector<double> &scores,
                           bool relax,
                           vector<double> *predicted_output);

    void DecodeLabel(int slen,
                     const vector<int> &pred_indices,
                     const vector<unordered_map<PKey, int>> &arg_indices_by_predicate,
                     const vector<set<int>> &roles_by_predicate,
                     const vector<double> &scores, vector<bool> &selected_parts, double *value);

    void DecodeLabelPredicate(int slen, const set<int> &roles,
                              const unordered_map<PKey, int> &indices,
                              const vector<double> &scores,
                              vector<bool> &selected_parts, double *value);

protected:

    void BuildLabelIndices(Instance *instance, Parts *parts, AD3::FactorGraph *factor_graph,
                           vector<int> *part_indices,
                           vector<AD3::BinaryVariable *> *variables,
                           vector<int> &pred_indices,
                           vector<unordered_map<PKey, int>> &arg_indices_by_predicate,
                           vector<set<int>> &roles_by_predicate,
                           const vector<double> &scores);

public:
    // Not implemented
    void DecodeCostAugmentedMarginals(Instance *instance, Parts *parts,
                                      const vector<double> &scores,
                                      const vector<double> &gold_output,
                                      vector<double> *predicted_output,
                                      double *entropy,
                                      double *cost,
                                      double *loss) {
        CHECK(false) << "Not implemented yet.";
    }

    void DecodeMarginals(Instance *instance, Parts *parts,
                         const vector<double> &scores,
                         const vector<double> &gold_output,
                         vector<double> *predicted_output,
                         double *entropy,
                         double *loss) {
        CHECK(false) <<"Not implemented.";
    }

protected:
    SemanticPipe *pipe_;
};

#endif /* SEMANTICDECODER_H_ */
