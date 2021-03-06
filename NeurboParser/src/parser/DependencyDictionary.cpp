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

#include "DependencyDictionary.h"
#include "DependencyPipe.h"
#include <algorithm>

void DependencyDictionary::CreateLabelDictionary(DependencyReader *reader) {
    LOG(INFO) << "Creating label dictionary...";

    vector<int> label_freqs;

    // Go through the corpus and build the label dictionary,
    // counting the frequencies.
    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    DependencyInstance *instance =
            static_cast<DependencyInstance *>(reader->GetNext());
    while (instance != NULL) {
        int instance_length = instance->size();
        for (int i = 1; i < instance_length; ++i) {
            int id;

            // Add dependency label to alphabet.
            id = label_alphabet_.Insert(instance->GetDependencyRelation(i));
            if (id >= label_freqs.size()) {
                CHECK_EQ(id, label_freqs.size());
                label_freqs.push_back(0);
            }
            ++label_freqs[id];
        }
        delete instance;
        instance = static_cast<DependencyInstance *>(reader->GetNext());
    }
    reader->Close();
    label_alphabet_.StopGrowth();

    // Go through the corpus and build the existing labels for each head-modifier
    // POS pair.
    existing_labels_.clear();
    existing_labels_.resize(token_dictionary_->GetNumPosTags(),
                            vector<vector<int> >(
                                    token_dictionary_->GetNumPosTags()));

    maximum_left_distances_.clear();
    maximum_left_distances_.resize(token_dictionary_->GetNumPosTags(),
                                   vector<int>(
                                           token_dictionary_->GetNumPosTags(), 0));

    maximum_right_distances_.clear();
    maximum_right_distances_.resize(token_dictionary_->GetNumPosTags(),
                                    vector<int>(
                                            token_dictionary_->GetNumPosTags(), 0));

    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    instance = static_cast<DependencyInstance *>(reader->GetNext());
    while (instance != NULL) {
        int instance_length = instance->size();
        for (int i = 1; i < instance_length; ++i) {
            int id;
            int head = instance->GetHead(i);
            CHECK_GE(head, 0);
            CHECK_LT(head, instance_length);
            const string &modifier_pos = instance->GetPosTag(i);
            const string &head_pos = instance->GetPosTag(head);
            int modifier_pos_id = token_dictionary_->GetPosTagId(modifier_pos);
            int head_pos_id = token_dictionary_->GetPosTagId(head_pos);
            if (modifier_pos_id < 0) modifier_pos_id = TOKEN_UNKNOWN;
            if (head_pos_id < 0) head_pos_id = TOKEN_UNKNOWN;
            //CHECK_GE(modifier_pos_id, 0);
            //CHECK_GE(head_pos_id, 0);

            id = label_alphabet_.Lookup(instance->GetDependencyRelation(i));
            CHECK_GE(id, 0);

            // Insert new label in the set of existing labels, if it is not there
            // already. NOTE: this is inefficient, maybe we should be using a
            // different data structure.
            vector<int> &labels = existing_labels_[modifier_pos_id][head_pos_id];
            int j;
            for (j = 0; j < labels.size(); ++j) {
                if (labels[j] == id) break;
            }
            if (j == labels.size()) labels.push_back(id);

            // Update the maximum distances if necessary.
            if (head != 0) {
                if (head < i) {
                    // Right attachment.
                    if (i - head > maximum_right_distances_[modifier_pos_id][head_pos_id]) {
                        maximum_right_distances_[modifier_pos_id][head_pos_id] = i - head;
                    }
                } else {
                    // Left attachment.
                    if (head - i > maximum_left_distances_[modifier_pos_id][head_pos_id]) {
                        maximum_left_distances_[modifier_pos_id][head_pos_id] = head - i;
                    }
                }
            }
        }
        delete instance;
        instance = static_cast<DependencyInstance *>(reader->GetNext());
    }
    reader->Close();

    LOG(INFO) << "Number of labels: " << label_alphabet_.size();
}

void DependencyTokenDictionary::Initialize(DependencyReader *reader) {
    SetTokenDictionaryFlagValues();
    LOG(INFO) << "Creating token dictionary...";

    vector<int> form_freqs;
    vector<int> form_lower_freqs;
    vector<int> lemma_freqs;
    vector<int> feats_freqs;
    vector<int> pos_freqs;
    vector<int> cpos_freqs;

    Alphabet form_alphabet;
    Alphabet form_lower_alphabet;
    Alphabet lemma_alphabet;
    Alphabet feats_alphabet;
    Alphabet pos_alphabet;
    Alphabet cpos_alphabet;

    string special_symbols[NUM_SPECIAL_TOKENS];
    special_symbols[TOKEN_UNKNOWN] = kTokenUnknown;
    special_symbols[TOKEN_START] = kTokenStart;
    special_symbols[TOKEN_STOP] = kTokenStop;

    for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
        prefix_alphabet_.Insert(special_symbols[i]);
        suffix_alphabet_.Insert(special_symbols[i]);
        form_alphabet.Insert(special_symbols[i]);
        form_lower_alphabet.Insert(special_symbols[i]);
        lemma_alphabet.Insert(special_symbols[i]);
        feats_alphabet.Insert(special_symbols[i]);
        pos_alphabet.Insert(special_symbols[i]);
        cpos_alphabet.Insert(special_symbols[i]);

        // Counts of special symbols are set to -1:
        form_freqs.push_back(-1);
        form_lower_freqs.push_back(-1);
        lemma_freqs.push_back(-1);
        feats_freqs.push_back(-1);
        pos_freqs.push_back(-1);
        cpos_freqs.push_back(-1);
    }

    // Go through the corpus and build the dictionaries,
    // counting the frequencies.
    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    DependencyInstance *instance =
            static_cast<DependencyInstance *>(reader->GetNext());
    while (instance != NULL) {
        int instance_length = instance->size();
        for (int i = 0; i < instance_length; ++i) {
            int id;

            // Add form to alphabet.
            std::string form = instance->GetForm(i);
            std::string form_lower(form);
            transform(form_lower.begin(), form_lower.end(),
                      form_lower.begin(), ::tolower);
            if (!form_case_sensitive) form = form_lower;
            id = form_alphabet.Insert(form);
            if (id >= form_freqs.size()) {
                CHECK_EQ(id, form_freqs.size());
                form_freqs.push_back(0);
            }
            ++form_freqs[id];

            // Add lower-case form to alphabet.
            id = form_lower_alphabet.Insert(form_lower);
            if (id >= form_lower_freqs.size()) {
                CHECK_EQ(id, form_lower_freqs.size());
                form_lower_freqs.push_back(0);
            }
            ++form_lower_freqs[id];

            // Add lemma to alphabet.
            id = lemma_alphabet.Insert(instance->GetLemma(i));
            if (id >= lemma_freqs.size()) {
                CHECK_EQ(id, lemma_freqs.size());
                lemma_freqs.push_back(0);
            }
            ++lemma_freqs[id];

            // Add prefix/suffix to alphabet.
            // TODO: add varying lengths.
            string prefix = form.substr(0, prefix_length);
            id = prefix_alphabet_.Insert(prefix);
            int start = form.length() - suffix_length;
            if (start < 0) start = 0;
            string suffix = form.substr(start, suffix_length);
            id = suffix_alphabet_.Insert(suffix);

            // Add POS to alphabet.
            id = pos_alphabet.Insert(instance->GetPosTag(i));
            if (id >= pos_freqs.size()) {
                CHECK_EQ(id, pos_freqs.size());
                pos_freqs.push_back(0);
            }
            ++pos_freqs[id];

            // Add CPOS to alphabet.
            id = cpos_alphabet.Insert(instance->GetCoarsePosTag(i));
            if (id >= cpos_freqs.size()) {
                CHECK_EQ(id, cpos_freqs.size());
                cpos_freqs.push_back(0);
            }
            ++cpos_freqs[id];

            // Add FEATS to alphabet.
            for (int j = 0; j < instance->GetNumMorphFeatures(i); ++j) {
                id = feats_alphabet.Insert(instance->GetMorphFeature(i, j));
                if (id >= feats_freqs.size()) {
                    CHECK_EQ(id, feats_freqs.size());
                    feats_freqs.push_back(0);
                }
                ++feats_freqs[id];
            }
        }
        delete instance;
        instance = static_cast<DependencyInstance *>(reader->GetNext());
    }
    reader->Close();

    // Now adjust the cutoffs if necessary.
    while (true) {
        form_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            form_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = form_alphabet.begin();
             iter != form_alphabet.end();
             ++iter) {
            if (form_freqs[iter->second] > form_cutoff) {
                form_alphabet_.Insert(iter->first);
            }
        }
        if (form_alphabet_.size() < kMaxFormAlphabetSize) break;
        ++form_cutoff;
        LOG(INFO) << "Incrementing form cutoff to " << form_cutoff << "...";
    }

    while (true) {
        form_lower_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            form_lower_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = form_lower_alphabet.begin();
             iter != form_lower_alphabet.end();
             ++iter) {
            if (form_lower_freqs[iter->second] > form_lower_cutoff) {
                form_lower_alphabet_.Insert(iter->first);
            }
        }
        if (form_lower_alphabet_.size() < kMaxFormAlphabetSize) break;
        ++form_lower_cutoff;
        LOG(INFO) << "Incrementing lower-case form cutoff to "
                  << form_lower_cutoff << "...";
    }

    while (true) {
        lemma_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            lemma_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = lemma_alphabet.begin();
             iter != lemma_alphabet.end();
             ++iter) {
            if (lemma_freqs[iter->second] > lemma_cutoff) {
                lemma_alphabet_.Insert(iter->first);
            }
        }
        if (lemma_alphabet_.size() < kMaxLemmaAlphabetSize) break;
        ++lemma_cutoff;
        LOG(INFO) << "Incrementing lemma cutoff to " << lemma_cutoff << "...";
    }

    while (true) {
        pos_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            pos_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = pos_alphabet.begin();
             iter != pos_alphabet.end();
             ++iter) {
            if (pos_freqs[iter->second] > pos_cutoff) {
                pos_alphabet_.Insert(iter->first);
            }
        }
        if (pos_alphabet_.size() < kMaxPosAlphabetSize) break;
        ++pos_cutoff;
        LOG(INFO) << "Incrementing POS cutoff to " << pos_cutoff << "...";
    }

    while (true) {
        cpos_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            cpos_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = cpos_alphabet.begin();
             iter != cpos_alphabet.end();
             ++iter) {
            if (cpos_freqs[iter->second] > cpos_cutoff) {
                cpos_alphabet_.Insert(iter->first);
            }
        }
        if (cpos_alphabet_.size() < kMaxCoarsePosAlphabetSize) break;
        ++cpos_cutoff;
        LOG(INFO) << "Incrementing CPOS cutoff to " << cpos_cutoff << "...";
    }

    while (true) {
        feats_alphabet_.clear();
        for (int i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
            feats_alphabet_.Insert(special_symbols[i]);
        }
        for (Alphabet::iterator iter = feats_alphabet.begin();
             iter != feats_alphabet.end();
             ++iter) {
            if (feats_freqs[iter->second] > feats_cutoff) {
                feats_alphabet_.Insert(iter->first);
            }
        }
        if (feats_alphabet_.size() < kMaxFeatsAlphabetSize) break;
        ++feats_cutoff;
        LOG(INFO) << "Incrementing FEATS cutoff to " << feats_cutoff << "...";
    }

    form_alphabet_.StopGrowth();
    form_lower_alphabet_.StopGrowth();
    lemma_alphabet_.StopGrowth();
    prefix_alphabet_.StopGrowth();
    suffix_alphabet_.StopGrowth();
    feats_alphabet_.StopGrowth();
    pos_alphabet_.StopGrowth();
    cpos_alphabet_.StopGrowth();

    LOG(INFO) << "Number of forms: " << form_alphabet_.size() << endl
              << "Number of lower-case forms: " << form_lower_alphabet_.size() << endl
              << "Number of lemmas: " << lemma_alphabet_.size() << endl
              << "Number of prefixes: " << prefix_alphabet_.size() << endl
              << "Number of suffixes: " << suffix_alphabet_.size() << endl
              << "Number of feats: " << feats_alphabet_.size() << endl
              << "Number of pos: " << pos_alphabet_.size() << endl
              << "Number of cpos: " << cpos_alphabet_.size();

    CHECK_LT(form_alphabet_.size(), 0xffff);
    CHECK_LT(form_lower_alphabet_.size(), 0xffff);
    CHECK_LT(lemma_alphabet_.size(), 0xffff);
    CHECK_LT(prefix_alphabet_.size(), 0xffff);
    CHECK_LT(suffix_alphabet_.size(), 0xffff);
    CHECK_LT(feats_alphabet_.size(), 0xffff);
    CHECK_LT(pos_alphabet_.size(), 0xff);
    CHECK_LT(cpos_alphabet_.size(), 0xff);

    // TODO: Remove this (only for debugging purposes).
    BuildNames();
}
