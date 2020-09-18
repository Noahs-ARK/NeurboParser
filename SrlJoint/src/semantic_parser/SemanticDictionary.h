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

#ifndef SEMANTICDICTIONARY_H_
#define SEMANTICDICTIONARY_H_

#include "Dictionary.h"
#include "TokenDictionary.h"
#include "DependencyDictionary.h"
#include "SerializationUtils.h"
#include "SemanticPredicate.h"
#include "SemanticReader.h"


const int bio_cutoff = 0;

class Pipe;

const string kRoleUnknown = "_UNKROLE_";
const string kPredicateUnknown = "_UNKPREDICATE_";
const string kRoleNone = "_NONE_";
const unsigned int IndexRoleNone = 1;

enum SpecialPredicates {
    PREDICATE_UNKNOWN = 0,
    NUM_SPECIAL_PREDICATES
};

enum BIOTags {
    BIO_S = 0,
    BIO_B,
    BIO_I,
    BIO_O,
    NUM_BIO_TAGS
};
class SemanticDictionary : public Dictionary {
public:
    SemanticDictionary() { token_dictionary_ = NULL; }

    SemanticDictionary(Pipe *pipe) : pipe_(pipe) {}

    virtual ~SemanticDictionary() {
        Clear();
    }

    void CreateFrameDictionaries(Alphabet &lu_alphabet,
                                 Alphabet &lu_name_alphabet,
                                 Alphabet &lu_pos_alphabet,
                                 Alphabet &frame_alphabet,
                                 Alphabet &role_alphabet);

    void CreatePredicateRoleDictionaries(SemanticReader *reader);

    void Clear() {
        // Don't clear token_dictionary, since this class does not own it.
//        for (int i = 0; i < name_predicates_.size(); ++i) {
//            for (int j = 0; j < name_predicates_[i].size(); ++j) {
//                delete name_predicates_[i][j];
//            }
//            name_predicates_[i].clear();
//        }
//        name_predicates_.clear();
        lu_alphabet_.clear();
        lu_name_alphabet_.clear();
        lu_pos_alphabet_.clear();
        frame_alphabet_.clear();
        role_alphabet_.clear();
        existing_roles_.clear();
        core_roles_.clear();
        frame_by_lu_.clear();
        existing_bios_.clear();
        role_maximum_length_.clear();
        maximum_left_distances_.clear();
        maximum_right_distances_.clear();
    }

    void BuildPredicateRoleNames() {
        lu_name_alphabet_.BuildNames();
        lu_pos_alphabet_.BuildNames();
        frame_alphabet_.BuildNames();
        role_alphabet_.BuildNames();
    }

//    const vector<SemanticPredicate *> &GetNamePredicates(int predicate_name) const {
//        return name_predicates_[predicate_name];
//    }

    const string &GetLu(int p) const {
        return lu_alphabet_.GetName(p);
    }

    const string &GetLuName(int p) const {
        return lu_name_alphabet_.GetName(p);
    }

    const string &GetLuPos(int p) const {
        return lu_pos_alphabet_.GetName(p);
    }

    const string &GetFrame(int p) const {
        return frame_alphabet_.GetName(p);
    }

    const string &GetRoleName(int role) const {
        return role_alphabet_.GetName(role);
    }

    int GetRoleBigramLabel(int first_role, int second_role) const {
        CHECK_GE(first_role, 0);
        CHECK_GE(second_role, 0);
        return first_role * role_alphabet_.size() + second_role;
    }

    int GetNumRoleBigramLabels() const {
        return role_alphabet_.size() * role_alphabet_.size();
    }

    int GetNumRoles() const {
        return role_alphabet_.size();
    }

    bool IsRoleDeterministic(int role) const {
        return deterministic_roles_[role];
    }

    bool IsCoreRole(int frame, int role) const {
        for (int i = 0;i < core_roles_[frame].size(); ++ i) {
            if (core_roles_[frame][i] == role) return true;
        }
        return false;
    }

    // TODO(atm): check if we should allow/stop growth of the other dictionaries
    // as well.
    void AllowGrowth() { token_dictionary_->AllowGrowth(); }

    void StopGrowth() { token_dictionary_->StopGrowth(); }

    void Save(FILE *fs) {
        if (0 > lu_alphabet_.Save(fs)) CHECK(false);
        if (0 > lu_name_alphabet_.Save(fs)) CHECK(false);
        if (0 > lu_pos_alphabet_.Save(fs)) CHECK(false);
        if (0 > frame_alphabet_.Save(fs)) CHECK(false);
        if (0 > role_alphabet_.Save(fs)) CHECK(false);
        bool success;
//        int length = name_predicates_.size();
//        success = WriteInteger(fs, length);
//        CHECK(success);
//        for (int i = 0; i < name_predicates_.size(); ++i) {
//            length = name_predicates_[i].size();
//            success = WriteInteger(fs, length);
//            CHECK(success);
//            for (int j = 0; j < name_predicates_[i].size(); ++j) {
//                name_predicates_[i][j]->Save(fs);
//            }
//        }
        CHECK_EQ(deterministic_roles_.size(), GetNumRoles());
        int length = deterministic_roles_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < deterministic_roles_.size(); ++i) {
            bool deterministic = deterministic_roles_[i];
            success = WriteBool(fs, deterministic);
            CHECK(success);
        }
        length = existing_roles_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < existing_roles_.size(); ++i) {
            length = existing_roles_[i].size();
            success = WriteInteger(fs, length);
            CHECK(success);
            for (int j = 0; j < existing_roles_[i].size(); ++j) {
                int role = existing_roles_[i][j];
                success = WriteInteger(fs, role);
                CHECK(success);
            }
        }
        length = core_roles_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < core_roles_.size(); ++i) {
            length = core_roles_[i].size();
            success = WriteInteger(fs, length);
            CHECK(success);
            for (int j = 0; j < core_roles_[i].size(); ++j) {
                int role = core_roles_[i][j];
                success = WriteInteger(fs, role);
                CHECK(success);
            }
        }
        length = frame_by_lu_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < frame_by_lu_.size(); ++i) {
            length = frame_by_lu_[i].size();
            success = WriteInteger(fs, length);
            CHECK(success);
            for (int j = 0; j < frame_by_lu_[i].size(); ++j) {
                int role = frame_by_lu_[i][j];
                success = WriteInteger(fs, role);
                CHECK(success);
            }
        }
        length = existing_bios_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < existing_bios_.size(); ++i) {
            length = existing_bios_[i].size();
            success = WriteInteger(fs, length);
            CHECK(success);
            for (int j = 0; j < existing_bios_[i].size(); ++j) {
                int bio = existing_bios_[i][j];
                success = WriteInteger(fs, bio);
                CHECK(success);
            }
        }
        length = role_maximum_length_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < role_maximum_length_.size(); ++ i) {
            int span_len = role_maximum_length_[i];
            success = WriteInteger(fs, span_len);
            CHECK(success);
        }
        length = maximum_left_distances_.size();
        success = WriteInteger(fs, length);
        CHECK(success);
        for (int i = 0; i < maximum_left_distances_.size(); ++i) {
            length = maximum_left_distances_[i].size();
            success = WriteInteger(fs, length);
            CHECK(success);
            for (int j = 0; j < maximum_left_distances_[i].size(); ++j) {
                int distance;
                distance = maximum_left_distances_[i][j];
                success = WriteInteger(fs, distance);
                CHECK(success);
                distance = maximum_right_distances_[i][j];
                success = WriteInteger(fs, distance);
                CHECK(success);
            }
        }
    }

    void Load(FILE *fs) {
        if (0 > lu_alphabet_.Load(fs)) CHECK(false);
        if (0 > lu_name_alphabet_.Load(fs)) CHECK(false);
        if (0 > lu_pos_alphabet_.Load(fs)) CHECK(false);
        if (0 > frame_alphabet_.Load(fs)) CHECK(false);
        if (0 > role_alphabet_.Load(fs)) CHECK(false);
        bool success;
        int length;
//        success = ReadInteger(fs, &length);
//        CHECK(success);
//        name_predicates_.resize(length);
//        for (int i = 0; i < name_predicates_.size(); ++i) {
//            success = ReadInteger(fs, &length);
//            CHECK(success);
//            name_predicates_[i].resize(length);
//            for (int j = 0; j < name_predicates_[i].size(); ++j) {
//                name_predicates_[i][j] = new SemanticPredicate();
//                name_predicates_[i][j]->Load(fs);
//            }
//        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        deterministic_roles_.resize(length);
        CHECK_EQ(deterministic_roles_.size(), GetNumRoles());
        for (int i = 0; i < deterministic_roles_.size(); ++i) {
            bool deterministic;
            success = ReadBool(fs, &deterministic);
            CHECK(success);
            deterministic_roles_[i] = deterministic;
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        existing_roles_.resize(length);
        for (int i = 0; i < existing_roles_.size(); ++i) {
            success = ReadInteger(fs, &length);
            CHECK(success);
            existing_roles_[i].resize(length);
            for (int j = 0; j < existing_roles_[i].size(); ++j) {
                int role;
                success = ReadInteger(fs, &role);
                CHECK(success);
                existing_roles_[i][j] = role;
            }
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        core_roles_.resize(length);
        for (int i = 0; i < core_roles_.size(); ++i) {
            success = ReadInteger(fs, &length);
            CHECK(success);
            core_roles_[i].resize(length);
            for (int j = 0; j < core_roles_[i].size(); ++j) {
                int role;
                success = ReadInteger(fs, &role);
                CHECK(success);
                core_roles_[i][j] = role;
            }
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        frame_by_lu_.resize(length);
        for (int i = 0; i < frame_by_lu_.size(); ++i) {
            success = ReadInteger(fs, &length);
            CHECK(success);
            frame_by_lu_[i].resize(length);
            for (int j = 0; j < frame_by_lu_[i].size(); ++j) {
                int role;
                success = ReadInteger(fs, &role);
                CHECK(success);
                frame_by_lu_[i][j] = role;
            }
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        existing_bios_.resize(length);
        for (int i = 0; i < existing_bios_.size(); ++i) {
            success = ReadInteger(fs, &length);
            CHECK(success);
            existing_bios_[i].resize(length);
            for (int j = 0; j < existing_bios_[i].size(); ++j) {
                int bio;
                success = ReadInteger(fs, &bio);
                CHECK(success);
                existing_bios_[i][j] = bio;
            }
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        role_maximum_length_.resize(length);
        CHECK_EQ(role_maximum_length_.size(), GetNumRoles());
        for (int i = 0; i < role_maximum_length_.size(); ++i) {
            int span_len;
            success = ReadInteger(fs, &span_len);
            CHECK(success);
            role_maximum_length_[i] = span_len;
        }
        success = ReadInteger(fs, &length);
        CHECK(success);
        maximum_left_distances_.resize(length);
        maximum_right_distances_.resize(length);
        for (int i = 0; i < maximum_left_distances_.size(); ++i) {
            success = ReadInteger(fs, &length);
            CHECK(success);
            maximum_left_distances_[i].resize(length);
            maximum_right_distances_[i].resize(length);
            for (int j = 0; j < maximum_left_distances_[i].size(); ++j) {
                int distance;
                success = ReadInteger(fs, &distance);
                CHECK(success);
                maximum_left_distances_[i][j] = distance;
                success = ReadInteger(fs, &distance);
                CHECK(success);
                maximum_right_distances_[i][j] = distance;
            }
        }
        BuildPredicateRoleNames();
    }

    Pipe *GetPipe() const { return pipe_; }

    TokenDictionary *GetTokenDictionary() const { return token_dictionary_; }

    void SetTokenDictionary(TokenDictionary *token_dictionary) {
        token_dictionary_ = token_dictionary;
        //CHECK(token_dictionary_ == NULL);
    }

    const vector<int> &GetExistingRoles(int predicate_frame) const {
        return existing_roles_[predicate_frame];
    }

    const vector<int> &GetFramesByLu(int lu) const {
        return frame_by_lu_[lu];
    }

    bool GetExistingBio(int pos_tag, int bio_tag) {
        return existing_bios_[pos_tag][bio_tag] >= bio_cutoff;
    }

    int GetMaximumRoleLen(int role) { return role_maximum_length_[role]; }

    int GetMaximumLeftDistance(int predicate_pos_id, int argument_pos_id) {
        return maximum_left_distances_[predicate_pos_id][argument_pos_id];
    }

    int GetMaximumRightDistance(int predicate_pos_id, int argument_pos_id) {
        return maximum_right_distances_[predicate_pos_id][argument_pos_id];
    }

    const Alphabet &GetLuAlphabet() const { return lu_alphabet_; }

    const Alphabet &GetLuNameAlphabet() const { return lu_name_alphabet_; }

    const Alphabet &GetLuPOSAlphabet() const { return lu_pos_alphabet_; }

    const Alphabet &GetFrameAlphabet() const { return frame_alphabet_; }

    const Alphabet &GetRoleAlphabet() const { return role_alphabet_; };

protected:

protected:
    Pipe *pipe_;
    TokenDictionary *token_dictionary_;
//    vector<vector<SemanticPredicate *> > name_predicates_;
    Alphabet lu_alphabet_;
    Alphabet lu_name_alphabet_;
    Alphabet lu_pos_alphabet_;
    Alphabet frame_alphabet_;

    Alphabet role_alphabet_;
    vector<bool> deterministic_roles_;
    vector<vector<int>> existing_roles_;
    vector<vector<int>> core_roles_;
    vector<vector<int>> frame_by_lu_;
    vector<int> role_maximum_length_;
    vector<vector<int> > existing_bios_; // used for pruning based on pos tag.
    vector<vector<int> > maximum_left_distances_;
    vector<vector<int> > maximum_right_distances_;
};


class SemanticTokenDictionary : public TokenDictionary {
public:
    SemanticTokenDictionary() {};

    virtual ~SemanticTokenDictionary() {};

    void Initialize(SemanticReader *reader);
};

#endif /* SEMANTICDICTIONARY_H_ */
