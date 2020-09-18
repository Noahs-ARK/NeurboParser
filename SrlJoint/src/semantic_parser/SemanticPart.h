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

#ifndef SEMANTICPART_H_
#define SEMANTICPART_H_

#include <stdio.h>
#include <vector>
#include "Part.h"
#include <unordered_map>
#include <boost/functional/hash.hpp>

using namespace std;

enum {
    SEMANTICPART_PREDICATE = 0,
    SEMANTICPART_UNLABELEDARGUMENT,
    SEMANTICPART_ARGUMENT,
    NUM_SEMANTICPARTS
};

// Part for the event that a word is a predicate.
class SemanticPartPredicate : public Part {
public:
    SemanticPartPredicate() {
        t_ = lu_name_ = lu_pos_ = frame_ = start_index_ = end_index_ = -1;
    }

    SemanticPartPredicate(int target, int s_index, int e_index, int lu_name, int lu_pos, int frame) :
            t_(target), lu_name_(lu_name), lu_pos_(lu_pos),
            frame_(frame), start_index_(s_index), end_index_(e_index) {}

    virtual ~SemanticPartPredicate() {}

public:
    int target_index() const { return t_; }

    int lu_name() const { return lu_name_; }

    int lu_pos() const { return lu_pos_; }

    int frame() const { return frame_; }

    void span(int &s_index, int &e_index) const {
        s_index = start_index_;
        e_index = end_index_;
    }

public:
    int type() { return SEMANTICPART_PREDICATE; }

    bool operator==(const SemanticPartPredicate &p) const {
        int s, e;
        p.span(s, e);
        return (start_index_ == s
                && end_index_ == e
                && frame_ == p.frame()
                && lu_name_ == p.lu_name()
                && lu_pos_ == p.lu_pos());
    }

private:
    int t_; // Index of the target.
    int lu_name_;
    int lu_pos_;
    int frame_; // frame id
    int start_index_; // token span
    int end_index_;
};

struct PredicatePartHasher {
    std::size_t operator()(const SemanticPartPredicate &p) const {
        using boost::hash_value;
        using boost::hash_combine;

        std::size_t seed = 0;
        int s, e;
        p.span(s, e);
        hash_combine(seed, hash_value(s));
        hash_combine(seed, hash_value(e));
        hash_combine(seed, hash_value(p.lu_name()));
        hash_combine(seed, hash_value(p.lu_pos()));
        hash_combine(seed, hash_value(p.frame()));
        return seed;
    }
};

// Part for the event that a word is an argument of at least one predicate.
class SemanticPartArgument : public Part {
public:
    SemanticPartArgument() { pred_idx_ = frame_ = r_ = start_index_ = end_index_ = -1; }

    SemanticPartArgument(int pred_idx, int frame, int s_index, int e_index, int role) :
            pred_idx_(pred_idx), start_index_(s_index), end_index_(e_index), frame_(frame), r_(role) {}

    virtual ~SemanticPartArgument() {}

    int role() const { return r_; }

    int pred_idx() const { return pred_idx_; }

    int frame() const { return frame_; }

    void span(int &s_index, int &e_index) const {
        s_index = start_index_;
        e_index = end_index_;
    }

    int type() { return SEMANTICPART_ARGUMENT; }

    bool operator==(const SemanticPartArgument &a) const {
        int as, ae;
        a.span(as, ae);
        return (start_index_ == as
                && end_index_ == ae
                && r_ == a.role()
                && frame_ == a.frame());
    }

private:
    int pred_idx_; // predicate part idx
    int frame_; // frame id
    int r_; // semantic role
    int start_index_; // token span
    int end_index_;
};


struct ArgumentPartHasher {
    std::size_t operator()(const SemanticPartArgument &a) const
    {
        using boost::hash_value;
        using boost::hash_combine;

        std::size_t seed = 0;
        int as, ae;
        a.span(as, ae);
        hash_combine(seed, hash_value(as));
        hash_combine(seed, hash_value(ae));
        hash_combine(seed, hash_value(a.role()));
        hash_combine(seed, hash_value(a.frame()));
        return seed;
    }
};


class SemanticParts : public Parts {
public:
    SemanticParts() { };

    virtual ~SemanticParts() { DeleteAll(); };

    void Initialize() {
        DeleteAll();
        for (int i = 0; i < NUM_SEMANTICPARTS; ++i) {
            offsets_[i] = -1;
        }
        for (int r = 0; r < argument_by_predicate_.size(); ++r) {
            argument_by_predicate_[r].clear();
        }
        argument_by_predicate_.clear();
    }

    Part *CreatePartPredicate(int target, int start, int end, int lu_name, int lu_pos, int frame) {
        return new SemanticPartPredicate(target, start, end, lu_name, lu_pos, frame);
    }

    Part *CreatePartArgument(int pred_idx, int frame, int start, int end, int role) {
        return new SemanticPartArgument(pred_idx, frame, start, end, role);
    }

    int AddPredicatePart(Part *part) {
        int r = size();
        push_back(part);
        argument_by_predicate_.push_back(vector<int>(0));
        CHECK_EQ(size(), argument_by_predicate_.size());
        return r;
    }

    int AddArgumentPart(int pred_index, Part *part) {
        int r = size();
        push_back(part);
        CHECK_LT(pred_index, argument_by_predicate_.size());
        argument_by_predicate_[pred_index].push_back(r);
        return r;
    }

//    int AddPart(Part *part) {
//        int r = size();
//        push_back(part);
//        argument_by_predicate_.push_back(vector<int>(0));
//        //LOG(INFO) << "Adding part #" << r << " with type " << part->type();
//        CHECK_EQ(size(), argument_by_predicate_.size());
//        return r;
//    }

public:
    void DeleteAll();

public:
    void BuildIndices();

    void DeleteIndices();

    int FindPredicate(const SemanticPartPredicate &predicate) {
        auto it = predicate_index_.find(predicate);
        if (it != predicate_index_.end())
            return it->second;
        return -1;
    }

    int FindArgument(int p, const SemanticPartArgument &argument) {
        auto it = argument_index_[p].find(argument);
        if (it != argument_index_[p].end())
            return it->second;
        return -1;
    }

    const vector<int> &GetArgumentsByPredicate(int p) {
        CHECK_GE(p, 0);
        CHECK_LT(p, argument_by_predicate_.size());
        return argument_by_predicate_[p];
    }

    // Set/Get offsets:
    void ClearOffsets() {
        for (int i = 0; i < NUM_SEMANTICPARTS; ++i) {
            offsets_[i] = -1;
        }
    }

    void BuildOffsets() {
        for (int i = NUM_SEMANTICPARTS - 1; i >= 0; --i) {
            if (offsets_[i] < 0 || offsets_[i] > size()) {
                offsets_[i] = (i == NUM_SEMANTICPARTS - 1) ? size() : offsets_[i + 1];
            }
        }
    };

    void SetOffsetPredicate(int offset, int size) {
        SetOffset(SEMANTICPART_PREDICATE, offset, size);
    };

    void SetOffsetUnlabeledArgument(int offset, int size) {
        SetOffset(SEMANTICPART_UNLABELEDARGUMENT, offset, size);
    };

    void SetOffsetArgument(int offset, int size) {
        SetOffset(SEMANTICPART_ARGUMENT, offset, size);
    };

    void GetOffsetPredicate(int *offset, int *size) const {
        GetOffset(SEMANTICPART_PREDICATE, offset, size);
    };

    void GetOffsetUnlabeledArgument(int *offset, int *size) const {
        GetOffset(SEMANTICPART_UNLABELEDARGUMENT, offset, size);
    };

    void GetOffsetArgument(int *offset, int *size) const {
        GetOffset(SEMANTICPART_ARGUMENT, offset, size);
    };

private:
    // Get offset from part index.
    void GetOffset(int i, int *offset, int *size) const {
        *offset = offsets_[i];
        *size = (i < NUM_SEMANTICPARTS - 1) ? offsets_[i + 1] - (*offset) :
                SemanticParts::size() - (*offset);
    }

    // Set offset from part index.
    void SetOffset(int i, int offset, int size) {
        offsets_[i] = offset;
        if (i < NUM_SEMANTICPARTS - 1) offsets_[i + 1] = offset + size;
    }

private:
    unordered_map<SemanticPartPredicate, int, PredicatePartHasher> predicate_index_;
    unordered_map<int, unordered_map<SemanticPartArgument, int, ArgumentPartHasher>> argument_index_;
    vector<vector<int> > argument_by_predicate_;
    int offsets_[NUM_SEMANTICPARTS];
};

#endif /* SEMANTICPART_H_ */
