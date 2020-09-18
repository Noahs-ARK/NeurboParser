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

#ifndef SEMANTICINSTANCENUMERIC_H_
#define SEMANTICINSTANCENUMERIC_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include "Dictionary.h"
#include "SemanticInstance.h"
#include "SemanticDictionary.h"

using namespace std;

class PredicateNumeric : public Span {
public:
    PredicateNumeric (int s, int e, int lu_id, int lu_name_id, int lu_pos_id, int frame_id):
            Span(s, e), lu_id_(lu_id), lu_name_id_(lu_name_id), lu_pos_id_(lu_pos_id), frame_id_(frame_id) {}

    PredicateNumeric () {};

    virtual ~PredicateNumeric() {}

    int GetLu() const { return lu_id_; };

    int GetLuName() const { return lu_name_id_; }

    int GetLuPOS() const { return lu_pos_id_; }

    int GetFrame() const { return frame_id_; }

    bool operator==(const PredicateNumeric &p) const {
        int ps, pe;
        p.GetSpan(ps, pe);
        return (start_index_ == ps
                && end_index_ == pe
                && lu_id_ == p.GetLu()
                && lu_name_id_ == p.GetLuName()
                && frame_id_ == p.GetFrame());
    }
protected:
    int lu_id_;
    int lu_name_id_;
    int lu_pos_id_;
    int frame_id_;
};

struct PredicateHasher
{
    std::size_t operator()(const PredicateNumeric &p) const
    {
        using boost::hash_value;
        using boost::hash_combine;

        std::size_t seed = 0;
        int ps, pe;
        p.GetSpan(ps, pe);
        hash_combine(seed, hash_value(ps));
        hash_combine(seed, hash_value(pe));
        hash_combine(seed, hash_value(p.GetLu()));
        hash_combine(seed, hash_value(p.GetLuName()));
        hash_combine(seed, hash_value(p.GetLuPOS()));
        hash_combine(seed, hash_value(p.GetFrame()));
        return seed;
    }
};

class ArgumentNumeric : public Span {
public:
    ArgumentNumeric (int s, int e, int role_id):
            Span(s, e), role_id_(role_id) {}

    ArgumentNumeric() {};

    virtual ~ArgumentNumeric() {}

    int GetRoleId() const { return role_id_; }

    bool operator==(const ArgumentNumeric &a) const {
        int as, ae;
        a.GetSpan(as, ae);
        return (start_index_ == as
                && end_index_ == ae
                && role_id_ == a.GetRoleId());
    }

protected:
    int role_id_;
};

struct ArgumentHasher
{
    std::size_t operator()(const ArgumentNumeric &a) const
    {
        using boost::hash_value;
        using boost::hash_combine;

        std::size_t seed = 0;
        int as, ae;
        a.GetSpan(as, ae);
        hash_combine(seed, hash_value(as));
        hash_combine(seed, hash_value(ae));
        hash_combine(seed, hash_value(a.GetRoleId()));
        return seed;
    }
};


class SemanticInstanceNumeric : public Instance {
public:
    SemanticInstanceNumeric() {};

    virtual ~SemanticInstanceNumeric() { Clear(); };

    Instance *Copy() {
        CHECK(false) << "Not implemented.";
        return NULL;
    }

    int size() { return form_ids_.size(); };

    void Clear() {
        form_ids_.clear();
        form_lower_ids_.clear();
        lemma_ids_.clear();
        prefix_ids_.clear();
        suffix_ids_.clear();
        pos_ids_.clear();
        cpos_ids_.clear();
        is_noun_.clear();
        is_verb_.clear();
        is_punc_.clear();
        is_coord_.clear();
        predicates_.clear();
        for (int j = 0; j < arguments_.size(); ++j) {
            arguments_[j].clear();
        }
        arguments_.clear();

        index_predicates_.clear();
        for (int p = 0; p < index_arguments_.size(); ++p) {
            index_arguments_[p].clear();
        }
        index_arguments_.clear();
    }

    void Initialize(const SemanticDictionary &dictionary,
                    SemanticInstance *instance);

    void GetPredicateSpan(int t, int &start_idx, int &end_idx) {
        predicates_[t].GetSpan(start_idx, end_idx);
    }

    int GetNumPredicates() { return predicates_.size(); }

    int GetNumTarget() { return predicates_.size(); }

    int GetLu(int k) { return predicates_[k].GetLu(); }

    int GetLuName(int k) { return predicates_[k].GetLuName(); }

    int GetLuPos(int k) { return predicates_[k].GetLuPOS(); }

    int GetFrame(int k) {
        CHECK_LT(k, predicates_.size());
        return predicates_[k].GetFrame();
    }

    int GetNumArgumentsPredicate(int k) {
        CHECK_LT(k, arguments_.size());
        return arguments_[k].size();
    }

    void GetArgumentSpan(int k, int l, int &s, int &e) { arguments_[k][l].GetSpan(s, e); }

    int GetArgumentRoleId(int k, int l) { return arguments_[k][l].GetRoleId(); }

    void GetArgumentIndex(int k, int l, int &start_idx, int &end_idx) {
        arguments_[k][l].GetSpan(start_idx, end_idx);
    }

    int FindPredicate(const PredicateNumeric &p) {
        auto it = index_predicates_.find(p);
        if (it != index_predicates_.end())
            return it->second;
        return -1;
    }

    int FindArgument(int p, const ArgumentNumeric &a) {
        auto it = index_arguments_[p].find(a);
        if (it != index_arguments_[p].end())
            return it->second;
        return -1;
    }

    const vector<int> &GetFormIds() const { return form_ids_; }

    const vector<int> &GetFormLowerIds() const { return form_lower_ids_; }

    const vector<int> &GetLemmaIds() const { return lemma_ids_; }

    const vector<int> &GetPosIds() const { return pos_ids_; }

    const vector<int> &GetCoarsePosIds() const { return cpos_ids_; }

    int GetFormId(int i) { return form_ids_[i]; };

    int GetFormLowerId(int i) { return form_lower_ids_[i]; };

    int GetLemmaId(int i) { return lemma_ids_[i]; };

    int GetPrefixId(int i) { return prefix_ids_[i]; };

    int GetSuffixId(int i) { return suffix_ids_[i]; };

    int GetPosId(int i) { return pos_ids_[i]; };

    int GetCoarsePosId(int i) { return cpos_ids_[i]; };

    bool IsNoun(int i) { return is_noun_[i]; };

    bool IsVerb(int i) { return is_verb_[i]; };

    bool IsPunctuation(int i) { return is_punc_[i]; };

    bool IsCoordination(int i) { return is_coord_[i]; };

private:
    vector<int> form_ids_;
    vector<int> form_lower_ids_;
    vector<int> lemma_ids_;
    vector<int> prefix_ids_;
    vector<int> suffix_ids_;
    vector<int> pos_ids_;
    vector<int> cpos_ids_;
    vector<bool> is_noun_;
    vector<bool> is_verb_;
    vector<bool> is_punc_;
    vector<bool> is_coord_;

    vector<PredicateNumeric> predicates_;
    vector<vector<ArgumentNumeric> > arguments_;
    unordered_map<PredicateNumeric, int, PredicateHasher> index_predicates_;
    vector<unordered_map<ArgumentNumeric, int, ArgumentHasher>> index_arguments_;
};

#endif /* SEMANTICINSTANCENUMERIC_H_ */
