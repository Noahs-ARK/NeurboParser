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

#ifndef SEMANTICINSTANCE_H_
#define SEMANTICINSTANCE_H_

#include <string>
#include <vector>
#include "DependencyInstance.h"
#include <iostream>

class Span {
public:

    Span(int s, int e): start_index_(s), end_index_(e){}

    Span () {}

    virtual ~Span() {};

    void GetSpan(int &s, int &e) const {
        s = start_index_;
        e = end_index_;
    }

    bool operator==(const Span &s) const {
        return (start_index_ == s.start_index_
                && end_index_ == s.end_index_ );
    }

protected:
    int start_index_; // index of start token
    int end_index_; // index of end token, inclusive
};

class Predicate : public Span {
public:
    Predicate (int s, int e, string name, string frame):
            Span(s, e), name_(name), frame_(frame) {}

    Predicate () {};

    virtual ~Predicate() {}

    const string &GetName() const { return name_; }

    const string &GetFrame() const { return frame_; }

    bool operator==(const Predicate &p) const {
        int ps, pe;
        p.GetSpan(ps, pe);
        return (start_index_ == ps
                && end_index_ == pe
                && name_ == p.GetName()
                && frame_== p.GetFrame());
    }
protected:
    string name_;
    string frame_;
};

class Argument : public Span {
public:
    Argument (int s, int e, string role):
            Span(s, e), role_(role) {}

    Argument () {};

    virtual ~Argument() {}

    const string &GetRole() const { return role_; }

    bool operator==(const Argument &a) const {
        int as, ae;
        a.GetSpan(as, ae);
        return (start_index_ == as
                && end_index_ == ae
                && role_ == a.GetRole());
    }

protected:
    string role_;
};

class SemanticInstance : public Instance {
public:
    SemanticInstance() {};

    virtual ~SemanticInstance() {};

    Instance *Copy() {
        SemanticInstance *instance = new SemanticInstance();
        instance->Initialize(name_, forms_, lemmas_, cpostags_, postags_,
                             predicates_, arguments_);
        return static_cast<Instance *>(instance);
    }

    void Initialize(const string &name,
                    const vector<string> &forms,
                    const vector<string> &lemmas,
                    const vector<string> &cpos,
                    const vector<string> &pos,
                    const vector<Predicate> &predicates,
                    const vector<vector<Argument> > &arguments);


    int size() { return forms_.size(); };

    const string &GetForm(int i) { return forms_[i]; };

    const vector<string> &GetForms(int start_idx, int end_idx) {
        vector<string> forms;
        for (int i = start_idx; i <= end_idx; ++ i)
            forms.push_back(forms_[i]);
        return forms;
    }

    const string &GetLemma(int i) { return lemmas_[i]; };

    const vector<string> &GetLemma(int start_idx, int end_idx) {
        vector<string> lemmas;
        for (int i = start_idx; i <= end_idx; ++ i)
            lemmas.push_back(lemmas_[i]);
        return lemmas;
    }

    const string &GetCoarsePosTag(int i) { return cpostags_[i]; };

    const vector<string> &GetCoarsePosTag(int start_idx, int end_idx) {
        vector<string> cpostags;
        for (int i = start_idx; i <= end_idx; ++ i)
            cpostags.push_back(cpostags_[i]);
        return cpostags;
    }

    const string &GetPosTag(int i) { return postags_[i]; };

    const vector<string> &GetPosTag(int start_idx, int end_idx) {
        vector<string> postags;
        for (int i = start_idx; i <= end_idx; ++ i)
            postags.push_back(postags_[i]);
        return postags;
    }

    const string &GetName() { return name_; }

    int GetNumPredicates() { return predicates_.size(); }

    int GetNumTargets() { return predicates_.size(); }

    const string &GetPredicateName(int k) { return predicates_[k].GetName(); }

    const string &GetPredicateFrame(int k) { return predicates_[k].GetFrame(); }

    //int GetPredicateIndex(int k) { return predicate_indices_[k]; }

    void GetPredicateIndex(int k, int &start_idx, int &end_idx) {
        predicates_[k].GetSpan(start_idx, end_idx);
    }

    int GetNumArgumentsPredicate(int k) { return arguments_[k].size(); }

    const string &GetArgumentRole(int k, int l) { return arguments_[k][l].GetRole(); }

    //int GetArgumentIndex(int k, int l) { return argument_indices_[k][l]; }

    void GetArgumentIndex(int k, int l, int &start_idx, int &end_idx) {
        arguments_[k][l].GetSpan(start_idx, end_idx);
    }

    int GetNumArgument(int k) { return arguments_[k].size(); }

    const Argument &GetArgument(int k, int l) { return arguments_[k][l]; }

    void ClearPredicates() {
        predicates_.clear();
        for (int p = 0; p < arguments_.size(); ++p) {
            arguments_[p].clear();
        }
        arguments_.clear();
    }

    void AddPredicate(const Predicate &predicate,
                      const vector<Argument> arguments) {
        predicates_.push_back(predicate);
        arguments_.push_back(arguments);
    }

protected:
    // FORM: the forms - usually words, like "thought"
    vector<string> forms_;
    // LEMMA: the lemmas, or stems, e.g. "think"
    vector<string> lemmas_;
    // COURSE-POS: the course part-of-speech tags, e.g."V"
    vector<string> cpostags_;
    // FINE-POS: the fine-grained part-of-speech tags, e.g."VBD"
    vector<string> postags_;
    // Name of the sentence (e.g. "#2000001").
    string name_;
    // Names of the predicates (e.g. "take.01").
    vector<Predicate> predicates_;
    vector<vector<Argument>> arguments_;
};

#endif /* SEMANTICINSTANCE_H_*/
