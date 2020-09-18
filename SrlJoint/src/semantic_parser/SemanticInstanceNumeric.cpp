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

#include "SemanticInstanceNumeric.h"
#include "SemanticPipe.h"

using namespace std;

const int kUnknownPredicate = 0xffff;
const int kUnknownFrame = 0xffff;
const int kUnknownRole = 0xffff;
const int kUnknownRelationPath = 0xffff;
const int kUnknownPosPath = 0xffff;

void SemanticInstanceNumeric::Initialize(
        const SemanticDictionary &dictionary,
        SemanticInstance *instance) {
    TokenDictionary *token_dictionary = dictionary.GetTokenDictionary();
    SemanticOptions *options =
            static_cast<SemanticPipe *>(dictionary.GetPipe())->GetSemanticOptions();
    int length = instance->size();
    int i;
    int id;

    int prefix_length = FLAGS_prefix_length;
    int suffix_length = FLAGS_suffix_length;
    bool form_case_sensitive = FLAGS_form_case_sensitive;

    Clear();

    form_ids_.resize(length);
    form_lower_ids_.resize(length);
    lemma_ids_.resize(length);
    prefix_ids_.resize(length);
    suffix_ids_.resize(length);
    pos_ids_.resize(length);
    cpos_ids_.resize(length);
    //shapes_.resize(length);
    is_noun_.resize(length);
    is_verb_.resize(length);
    is_punc_.resize(length);
    is_coord_.resize(length);

    for (i = 0; i < length; i++) {
        std::string form = instance->GetForm(i);
        std::string form_lower(form);
        transform(form_lower.begin(), form_lower.end(), form_lower.begin(),
                  ::tolower);
        if (!form_case_sensitive) form = form_lower;
        id = token_dictionary->GetFormId(form);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_ids_[i] = id;

        id = token_dictionary->GetFormLowerId(form_lower);
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        form_lower_ids_[i] = id;

        id = token_dictionary->GetLemmaId(instance->GetLemma(i));
        CHECK_LT(id, 0xfffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        lemma_ids_[i] = id;

        std::string prefix = form.substr(0, prefix_length);
        id = token_dictionary->GetPrefixId(prefix);
        CHECK_LT(id, 0xffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        prefix_ids_[i] = id;

        int start = form.length() - suffix_length;
        if (start < 0) start = 0;
        std::string suffix = form.substr(start, suffix_length);
        id = token_dictionary->GetSuffixId(suffix);
        CHECK_LT(id, 0xffff);
        if (id < 0) id = TOKEN_UNKNOWN;
        suffix_ids_[i] = id;

        id = token_dictionary->GetPosTagId(instance->GetPosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        pos_ids_[i] = id;

        id = token_dictionary->GetCoarsePosTagId(instance->GetCoarsePosTag(i));
        CHECK_LT(id, 0xff);
        if (id < 0) id = TOKEN_UNKNOWN;
        cpos_ids_[i] = id;

        //GetWordShape(instance->GetForm(i), &shapes_[i]);

        // Check whether the word is a noun, verb, punctuation or coordination.
        // Note: this depends on the POS tag string.
        // This procedure is taken from EGSTRA
        // (http://groups.csail.mit.edu/nlp/egstra/).
        is_noun_[i] = false;
        is_verb_[i] = false;
        is_punc_[i] = false;
        is_coord_[i] = false;

        const char *tag = instance->GetPosTag(i).c_str();
        if (tag[0] == 'v' || tag[0] == 'V') {
            is_verb_[i] = true;
        } else if (tag[0] == 'n' || tag[0] == 'N') {
            is_noun_[i] = true;
        } else if (strcmp(tag, "Punc") == 0 ||
                   strcmp(tag, "$,") == 0 ||
                   strcmp(tag, "$.") == 0 ||
                   strcmp(tag, "PUNC") == 0 ||
                   strcmp(tag, "punc") == 0 ||
                   strcmp(tag, "F") == 0 ||
                   strcmp(tag, "IK") == 0 ||
                   strcmp(tag, "XP") == 0 ||
                   strcmp(tag, ",") == 0 ||
                   strcmp(tag, ";") == 0) {
            is_punc_[i] = true;
        } else if (strcmp(tag, "Conj") == 0 ||
                   strcmp(tag, "KON") == 0 ||
                   strcmp(tag, "conj") == 0 ||
                   strcmp(tag, "Conjunction") == 0 ||
                   strcmp(tag, "CC") == 0 ||
                   strcmp(tag, "cc") == 0) {
            is_coord_[i] = true;
        }
    }

    int num_predicates = instance->GetNumPredicates();
    predicates_.clear();
    arguments_.clear();
    index_arguments_.clear();
    int p_start, p_end, a_start, a_end;
    for (int k = 0; k < instance->GetNumPredicates(); k++) {
        const string &lu = instance->GetPredicateName(k);
        vector<string> lu_fields;
        StringSplit(lu, ".", &lu_fields, true);

        int lu_id = dictionary.GetLuAlphabet().Lookup(lu);
        int lu_name_id = dictionary.GetLuNameAlphabet().Lookup(lu_fields[0]);
        int lu_pos_id = dictionary.GetLuPOSAlphabet().Lookup(lu_fields[1]);
        CHECK_LT(lu_name_id, 0xffff);
        CHECK_LT(lu_pos_id, 0xffff);
        if (lu_name_id < 0) lu_name_id = dictionary.GetLuNameAlphabet().Lookup(kPredicateUnknown);
        if (lu_pos_id < 0) lu_pos_id = dictionary.GetLuPOSAlphabet().Lookup(kPredicateUnknown);

        const string &frame = instance->GetPredicateFrame(k);
        int frame_id = dictionary.GetFrameAlphabet().Lookup(frame);
        if (frame_id < 0)
            LOG(INFO) << frame;
        CHECK_LT(frame_id, 0xffff);
        if (frame_id < 0) continue; // TODO: Currently disallow unk frames.
        instance->GetPredicateIndex(k, p_start, p_end);
        CHECK_LE(p_start, p_end);
        PredicateNumeric predicate(p_start, p_end, lu_id, lu_name_id, lu_pos_id, frame_id);
        int r = predicates_.size();
        predicates_.push_back(predicate);
        index_predicates_[predicate] = k;
        int num_arguments = instance->GetNumArgumentsPredicate(k);

        arguments_.push_back(vector<ArgumentNumeric>(num_arguments));
        index_arguments_.push_back(unordered_map<ArgumentNumeric, int, ArgumentHasher>());
        vector<int> overt_roles;
        for (int l = 0; l < num_arguments; ++l) {
            const string &role = instance->GetArgumentRole(k, l);
            int role_id = dictionary.GetRoleAlphabet().Lookup(role);
            CHECK_LT(role_id, 0xffff);
            if (role_id < 0) {
                role_id = dictionary.GetRoleAlphabet().Lookup(kRoleUnknown);
            }
            CHECK_GE(role_id, 0);
            overt_roles.push_back(role_id);
            instance->GetArgumentIndex(k, l, a_start, a_end);
            CHECK_LE(a_start, a_end);
            ArgumentNumeric argument(a_start, a_end, role_id);
            arguments_[r][l] = argument;
            index_arguments_[r][argument] = l;
        }
//        const vector<int> allowed_roles = dictionary.GetExistingRoles(frame_id);
//        for (int i = 0;i < allowed_roles.size(); ++ i) {
//            int role = allowed_roles[i];
//            bool role_overt = false;
//            for (int j = 0;j < overt_roles.size();++ j) {
//                if (overt_roles[j] == role) {
//                    role_overt = true;
//                    break;
//                }
//            }
//            if (!role_overt) {
//                // add an empty span if the role is not overt
//                ArgumentNumeric argument(-1, -1, role);
//                index_arguments_[r][argument] = arguments_[r].size();
//                arguments_[r].push_back(argument);
//            }
//        }
    }
    CHECK_EQ(predicates_.size(), arguments_.size());
    CHECK_EQ(predicates_.size(), index_arguments_.size());
}