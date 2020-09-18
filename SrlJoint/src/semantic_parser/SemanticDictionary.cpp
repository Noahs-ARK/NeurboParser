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

#include "SemanticDictionary.h"
#include "SemanticPipe.h"

// Special symbols.
//const string kPathUnknown = "_UNKNOWN_"; // Unknown path.

// Maximum alphabet sizes.
const unsigned int kMaxPredicateAlphabetSize = 0xffff;
const unsigned int kMaxRoleAlphabetSize = 0xffff;
const unsigned int kMaxRelationPathAlphabetSize = 0xffff;
const unsigned int kMaxPosPathAlphabetSize = 0xffff;
DEFINE_int32(role_cutoff, 1,
             "Ignore roles whose frequency is less than this.");
DEFINE_int32(lu_name_cutoff, 1,
             "Ignore lu names whose frequency is less than this.");
DEFINE_int32(lu_pos_cutoff, 1,
             "Ignore lu pos whose frequency is less than this.");

DEFINE_int32(relation_path_cutoff, 0,
             "Ignore relation paths whose frequency is less than this.");
DEFINE_int32(pos_path_cutoff, 0,
             "Ignore relation paths whose frequency is less than this.");
DEFINE_int32(num_frequent_role_pairs, 50,
             "Number of frequent role pairs to use in labeled sibling features");

void SemanticDictionary::CreateFrameDictionaries(Alphabet &lu_alphabet_tmp,
                                                 Alphabet &lu_name_alphabet_tmp,
                                                 Alphabet &lu_pos_alphabet_tmp,
                                                 Alphabet &frame_alphabet_tmp,
                                                 Alphabet &role_alphabet_tmp) {
    LOG(INFO) << "Creating frame dictionaries...";

    string special_symbols[NUM_SPECIAL_PREDICATES];
    special_symbols[PREDICATE_UNKNOWN] = kPredicateUnknown;
    for (int i = 0; i < NUM_SPECIAL_PREDICATES; ++i) {
        lu_alphabet_tmp.Insert(special_symbols[i]);
        lu_name_alphabet_tmp.Insert(special_symbols[i]);
        lu_pos_alphabet_tmp.Insert(special_symbols[i]);
        frame_alphabet_tmp.Insert(special_symbols[i]);
    }

    int index_role_unk = role_alphabet_tmp.Insert(kRoleUnknown);
    int index_role_none = role_alphabet_tmp.Insert(kRoleNone);

    ifstream is;
    string frame_file = static_cast<SemanticOptions *> (pipe_->GetOptions())->GetFrameFilePath();
    is.open(frame_file.c_str(), ifstream::in);
    CHECK(is.good()) << "Could not open " << frame_file << ".";
    string line_frame, line_roles, line_core_roles, line_lus;
    while (!is.eof()) {
        getline(is, line_frame);
        if (line_frame.length() == 0) continue;
        getline(is, line_roles);
        getline(is, line_core_roles);
        getline(is, line_lus);
        vector<string> frame, roles, core_roles, lus;
        StringSplit(line_frame, "\t", &frame, true);
        StringSplit(line_roles, "\t", &roles, true);
        StringSplit(line_core_roles, "\t", &core_roles, true);
        StringSplit(line_lus, "\t", &lus, true);
        CHECK_EQ(1, frame.size());

        int frame_id = frame_alphabet_tmp.Insert(frame[0]);
        for (int i = 0;i < roles.size(); ++ i) {
            int role_id = role_alphabet_tmp.Insert(roles[i]);
        }
        for (int i = 0;i < core_roles.size(); ++ i) {
            int role_id = role_alphabet_tmp.Lookup(core_roles[i]);
            CHECK_GE(role_id, 0);
        }
        for (int i = 0;i < lus.size(); ++ i) {
            vector<string> lu_fields;
            int lu_id = lu_alphabet_tmp.Insert(lus[i]);
            StringSplit(lus[i], ".", &lu_fields, true);
            CHECK_EQ(lu_fields.size(), 2);
            int lu_name_id = lu_name_alphabet_tmp.Insert(lu_fields[0]);
            int lu_pos_id = lu_pos_alphabet_tmp.Insert(lu_fields[1]);
        }
    }
    is.clear(); is.close();
    lu_alphabet_tmp.StopGrowth();
    lu_name_alphabet_tmp.StopGrowth();
    lu_pos_alphabet_tmp.StopGrowth();
    frame_alphabet_tmp.StopGrowth();
    role_alphabet_tmp.StopGrowth();

    CHECK_LT(lu_alphabet_tmp.size(), kMaxPredicateAlphabetSize);
    CHECK_LT(lu_name_alphabet_tmp.size(), kMaxPredicateAlphabetSize);
    CHECK_LT(lu_pos_alphabet_tmp.size(), kMaxPredicateAlphabetSize);
    CHECK_LT(frame_alphabet_tmp.size(), kMaxPredicateAlphabetSize);
    CHECK_LT(role_alphabet_tmp.size(), kMaxRoleAlphabetSize);
}

void SemanticDictionary::CreatePredicateRoleDictionaries(SemanticReader *reader) {

    CreateFrameDictionaries(lu_alphabet_, lu_name_alphabet_, lu_pos_alphabet_,
                            frame_alphabet_, role_alphabet_);
    LOG(INFO) << "Creating predicate and role dictionaries...";

    int num_lu_names = lu_name_alphabet_.size();
    vector<int> role_freqs;
    vector<int> lu_freqs;
    vector<int> lu_name_freqs;
    vector<int> lu_pos_freqs;
    vector<int> frame_freqs;


    int index_role_unk = role_alphabet_.Lookup(kRoleUnknown);
    int index_role_none = role_alphabet_.Lookup(kRoleNone);

    lu_freqs.assign(lu_alphabet_.size(), 0);
    lu_name_freqs.assign(lu_name_alphabet_.size(), 0);
    lu_pos_freqs.assign(lu_pos_alphabet_.size(), 0);
    frame_freqs.assign(frame_alphabet_.size(), 0);
    role_freqs.assign(role_alphabet_.size(), 0);

//    CHECK_EQ(index_role_none, IndexRoleNone);
    // Go through the corpus and build the predicate/roles dictionaries,
    // counting the frequencies.

    // TODO: find a cleaner way to get allowed roles.
    reader->Open(static_cast<SemanticOptions *> (pipe_->GetOptions())->GetTrainingFilePath());
    SemanticInstance *instance =
            static_cast<SemanticInstance *>(reader->GetNext());
    while (instance != NULL) {
        for (int k = 0; k < instance->GetNumPredicates(); ++k) {
            vector<string> lu_fields;
            const string lu = instance->GetPredicateName(k);
            StringSplit(lu, ".", &lu_fields, true);
            string lu_name = lu_fields[0];
            string lu_pos = lu_fields[1];
            const string frame = instance->GetPredicateFrame(k);

            int lu_id = lu_alphabet_.Lookup(lu);
            int lu_name_id = lu_name_alphabet_.Lookup(lu_name);
            int lu_pos_id = lu_pos_alphabet_.Lookup(lu_pos);
            int frame_id = frame_alphabet_.Lookup(frame);
            CHECK_GE(lu_name_id, 0);
            CHECK_GE(lu_pos_id, 0);
            CHECK_GE(frame_id, 0);
            ++ lu_name_freqs[lu_name_id];
            ++ lu_pos_freqs[lu_pos_id];
            ++ frame_freqs[frame_id];

            // Add semantic roles to alphabet.
            for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
                int role_id = role_alphabet_.Lookup(instance->GetArgumentRole(k, l));
                CHECK_GE(role_id, 0);
                ++ role_freqs[role_id];
            }
        }
        delete instance;
        instance = static_cast<SemanticInstance *>(reader->GetNext());
    }
    reader->Close();

    if (static_cast<SemanticOptions *> (pipe_->GetOptions())->use_exemplar()) {
        reader->Open(static_cast<SemanticOptions *> (pipe_->GetOptions())->GetExemplarFilePath());
        instance = static_cast<SemanticInstance *>(reader->GetNext());
        while (instance != NULL) {
            for (int k = 0; k < instance->GetNumPredicates(); ++k) {
                vector<string> lu_fields;
                const string lu = instance->GetPredicateName(k);
                StringSplit(lu, ".", &lu_fields, true);
                string lu_name = lu_fields[0];
                string lu_pos = lu_fields[1];
                const string frame = instance->GetPredicateFrame(k);

                int lu_id = lu_alphabet_.Lookup(lu);
                int lu_name_id = lu_name_alphabet_.Lookup(lu_name);
                int lu_pos_id = lu_pos_alphabet_.Lookup(lu_pos);
                int frame_id = frame_alphabet_.Lookup(frame);
                CHECK_GE(lu_name_id, 0);
                CHECK_GE(lu_pos_id, 0);
                CHECK_GE(frame_id, 0);

//                if (lu_name_freqs[lu_name_id] > 0 && frame_freqs[frame_id] == 0) {
//	                LOG(INFO) << lu_name <<" " << frame;
//                }

                ++ lu_name_freqs[lu_name_id];
                ++ lu_pos_freqs[lu_pos_id];
                ++ frame_freqs[frame_id];

                // Add semantic roles to alphabet.
                for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
                    int role_id = role_alphabet_.Lookup(instance->GetArgumentRole(k, l));
                    CHECK_GE(role_id, 0);
                    ++ role_freqs[role_id];
                }
            }
            delete instance;
            instance = static_cast<SemanticInstance *>(reader->GetNext());
        }
        reader->Close();
    }

    int role_cutoff = 1;

    LOG(INFO) << "Number of lu: " << lu_alphabet_.size();
    LOG(INFO) << "Number of lu name: " << lu_name_alphabet_.size();
    LOG(INFO) << "Number of lu pos: " << lu_pos_alphabet_.size();
    LOG(INFO) << "Number of frames: " << frame_alphabet_.size();
    LOG(INFO) << "Number of roles: " << role_alphabet_.size();

    existing_roles_.resize(frame_alphabet_.size());
    core_roles_.resize(frame_alphabet_.size());
    frame_by_lu_.resize(lu_alphabet_.size());

    ifstream is;
    string frame_file = static_cast<SemanticOptions *> (pipe_->GetOptions())->GetFrameFilePath();
    is.open(frame_file.c_str(), ifstream::in);
    CHECK(is.good()) << "Could not open " << frame_file << ".";
    string line_frame, line_roles, line_core_roles, line_lus;
    while (!is.eof()) {
        getline(is, line_frame);
        if (line_frame.length() == 0) continue;
        getline(is, line_roles);
        getline(is, line_core_roles);
        getline(is, line_lus);
        vector<string> frame, roles, core_roles, lus;
        StringSplit(line_frame, "\t", &frame, true);
        StringSplit(line_roles, "\t", &roles, true);
        StringSplit(line_core_roles, "\t", &core_roles, true);
        StringSplit(line_lus, "\t", &lus, true);
        CHECK_EQ(1, frame.size());

        // TODO: find a cleaner solution
        int frame_id = frame_alphabet_.Lookup(frame[0]);
        CHECK_GE(frame_id, 1);
        for (int i = 0;i < roles.size(); ++ i) {
            int role_id = role_alphabet_.Lookup(roles[i]);
            CHECK_GE(role_id, 2);
//            if (role_id < 0) role_id = index_role_unk;
            if (role_freqs[role_id] < role_cutoff) continue;//role_id = index_role_unk;
            int j = 0;
            for (j = 0;j < existing_roles_[frame_id].size(); ++ j) {
                if (existing_roles_[frame_id][j] == role_id) break;
            }
            if (j == existing_roles_[frame_id].size()) existing_roles_[frame_id].push_back(role_id);
        }
        for (int i = 0;i < core_roles.size(); ++ i) {
            int role_id = role_alphabet_.Lookup(core_roles[i]);
            if (role_id < 0) role_id = index_role_unk;
            int j = 0;
            for (j = 0;j < core_roles_[frame_id].size(); ++ j) {
                if (core_roles_[frame_id][j] == role_id) break;
            }
            if (j == core_roles_[frame_id].size()) core_roles_[frame_id].push_back(role_id);
        }

        for (int i = 0;i < lus.size(); ++ i) {
            int lu_id = lu_alphabet_.Lookup(lus[i]);
            CHECK_GE(lu_id, 0);
            int j = 0;
            for (j = 0;j < frame_by_lu_[lu_id].size(); ++ j) {
                if (frame_by_lu_[lu_id][j] == frame_id) break;
            }
            if (j == frame_by_lu_[lu_id].size()) frame_by_lu_[lu_id].push_back(frame_id);
        }
    }
    is.clear(); is.close();


    existing_bios_.assign(GetTokenDictionary()->GetNumPosTags(), vector<int>(NUM_BIO_TAGS, 0));

    vector<int> role_pair_freqs(GetNumRoleBigramLabels(), 0);
    // Initialize every label as deterministic.
    deterministic_roles_.assign(GetNumRoles(), true);
    role_maximum_length_.assign(GetNumRoles(), 0);

    maximum_left_distances_.clear();
    maximum_left_distances_.resize(token_dictionary_->GetNumPosTags(),
                                   vector<int>(
                                           token_dictionary_->GetNumPosTags(), 0));

    maximum_right_distances_.clear();
    maximum_right_distances_.resize(token_dictionary_->GetNumPosTags(),
                                    vector<int>(
                                            token_dictionary_->GetNumPosTags(), 0));
    // TODO: find a cleaner way to get allowed roles.
    reader->Open(static_cast<SemanticOptions *> (pipe_->GetOptions())->GetTrainingFilePath());
    instance = static_cast<SemanticInstance *>(reader->GetNext());
    int p_start_idx, p_end_idx;
    int a_start_idx, a_end_idx;
    while (instance != NULL) {
        for (int k = 0; k < instance->GetNumPredicates(); ++k) {
            instance->GetPredicateIndex(k, p_start_idx, p_end_idx);
            string frame = instance->GetPredicateFrame(k);
            int frame_id = frame_alphabet_.Lookup(frame);

            // Add semantic roles to alphabet.
            for (int l = 0; l < instance->GetNumArgumentsPredicate(k); ++l) {
                instance->GetArgumentIndex(k, l, a_start_idx, a_end_idx);
                int role_id = role_alphabet_.Lookup(instance->GetArgumentRole(k, l));
                if (role_id < 0)
                    role_id = index_role_unk;;
                if (a_end_idx - a_start_idx + 1 > role_maximum_length_[role_id])
                    role_maximum_length_[role_id] = a_end_idx - a_start_idx + 1;
                //CHECK_GE(role_id, 0);

                // Look for possible role pairs.
                for (int m = l + 1; m < instance->GetNumArgumentsPredicate(k); ++m) {
                    int other_role_id =
                            role_alphabet_.Lookup(instance->GetArgumentRole(k, m));
                    //CHECK_GE(other_role_id, 0);
                    if (other_role_id < 0)
                        other_role_id = index_role_unk;
                    int bigram_label = GetRoleBigramLabel(role_id, other_role_id);
                    CHECK_GE(bigram_label, 0);
                    CHECK_LT(bigram_label, GetNumRoleBigramLabels());
                    ++role_pair_freqs[bigram_label];
                    if (role_id == other_role_id) {
                        // Role label is not deterministic.
                        deterministic_roles_[role_id] = false;
                    }
                }

                // Insert new role in the set of existing labels, if it is not there
                // already. NOTE: this is inefficient, maybe we should be using a
                // different data structure.
                for (int p = p_start_idx; p <= p_end_idx; ++ p) {
                    string lu_pos = instance->GetPosTag(p);
                    int lu_pos_id = token_dictionary_->GetPosTagId(lu_pos);
                    for (int a = a_start_idx; a <= a_end_idx; ++ a) {
                        string argument_pos = instance->GetPosTag(a);
                        int argument_pos_id = token_dictionary_->GetPosTagId(argument_pos);
                        if (a_start_idx == a_end_idx) {
                            existing_bios_[argument_pos_id][BIO_S] += 1;
                        } else {
                            if (a == a_start_idx) existing_bios_[argument_pos_id][BIO_B] += 1;
                            else if (a == a_end_idx) existing_bios_[argument_pos_id][BIO_O] += 1;
                            else existing_bios_[argument_pos_id][BIO_I] += 1;
                        }
                        // Update the maximum distances if necessary.
                        if (p < a) {
                            // Right attachment.
                            if (a - p >
                                maximum_right_distances_[lu_pos_id][argument_pos_id]) {
                                maximum_right_distances_[lu_pos_id][argument_pos_id] = a - p;
                            }
                        } else {
                            // Left attachment (or self-loop). TODO(atm): treat self-loops differently?
                            if (p - a >
                                maximum_left_distances_[lu_pos_id][argument_pos_id]) {
                                maximum_left_distances_[lu_pos_id][argument_pos_id] = p - a;
                            }
                        }
                    }
                }
            }
        }
        delete instance;
        instance = static_cast<SemanticInstance *>(reader->GetNext());
    }
    reader->Close();


    // Compute the set of most frequent role pairs.
    vector<pair<int, int> > freqs_pairs;
    for (int k = 0; k < role_pair_freqs.size(); ++k) {
        freqs_pairs.push_back(pair<int, int>(-role_pair_freqs[k], k));
    }
    sort(freqs_pairs.begin(), freqs_pairs.end());

    // Display information about deterministic roles.
    int num_deterministic_roles = 0;
    for (Alphabet::iterator it = role_alphabet_.begin();
         it != role_alphabet_.end(); ++it) {
        string role = it->first;
        int role_id = it->second;
        if (IsRoleDeterministic(role_id)) {
//            LOG(INFO) << "Deterministic role: "
//                      << role;
            ++num_deterministic_roles;
        }
    }
    BuildPredicateRoleNames();
    LOG(INFO) << num_deterministic_roles << " out of "
              << GetNumRoles() << " roles are deterministic.";
}

void SemanticTokenDictionary::Initialize(SemanticReader *reader) {
    SetTokenDictionaryFlagValues();
    LOG(INFO) << "Creating token dictionary...";

    vector<int> form_freqs;
    vector<int> form_lower_freqs;
    vector<int> lemma_freqs;
    vector<int> pos_freqs;
    vector<int> cpos_freqs;

    Alphabet form_alphabet;
    Alphabet form_lower_alphabet;
    Alphabet lemma_alphabet;
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
        pos_alphabet.Insert(special_symbols[i]);
        cpos_alphabet.Insert(special_symbols[i]);

        // Counts of special symbols are set to -1:
        form_freqs.push_back(-1);
        form_lower_freqs.push_back(-1);
        lemma_freqs.push_back(-1);
        pos_freqs.push_back(-1);
        cpos_freqs.push_back(-1);
    }

    // Go through the corpus and build the dictionaries,
    // counting the frequencies.
//    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    reader->Open(pipe_->GetOptions()->GetTrainingFilePath());
    SemanticInstance *instance =
            static_cast<SemanticInstance *>(reader->GetNext());
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
	        std::string lemma = instance->GetLemma(i);
	        transform(lemma.begin(), lemma.end(),
	                  lemma.begin(), ::tolower);
            id = lemma_alphabet.Insert(lemma);
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
        }
        delete instance;
        instance = static_cast<SemanticInstance *>(reader->GetNext());
    }
    reader->Close();

    if (static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()->use_exemplar()) {
        reader->Open(static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()->GetExemplarFilePath());
        instance = static_cast<SemanticInstance *>(reader->GetNext());
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
	            std::string lemma = instance->GetLemma(i);
	            transform(lemma.begin(), lemma.end(),
	                      lemma.begin(), ::tolower);
	            id = lemma_alphabet.Insert(lemma);
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
            }
            delete instance;
            instance = static_cast<SemanticInstance *>(reader->GetNext());
        }
        reader->Close();
    }

	{
		ifstream in(static_cast<SemanticPipe *> (pipe_)->GetSemanticOptions()
				            ->GetPretrainedEmbeddingFilePath());

		if (!in.is_open()) {
			cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
		}
		string line;
		getline(in, line);
		string form;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> form;
			std::string form_lower(form);
			transform(form_lower.begin(), form_lower.end(),
			          form_lower.begin(), ::tolower);
			if (!form_case_sensitive) form = form_lower;
			int id = form_alphabet.Insert(form);
			if (id >= form_freqs.size()) {
				CHECK_EQ(id, form_freqs.size());
				form_freqs.push_back(0);
			}
			++ form_freqs[id];

			id = form_lower_alphabet.Insert(form);
			if (id >= form_lower_freqs.size()) {
				CHECK_EQ(id, form_lower_freqs.size());
				form_lower_freqs.push_back(0);
			}
			++form_lower_freqs[id];
		}
		in.close();
	}

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

    form_alphabet_.StopGrowth();
    form_lower_alphabet_.StopGrowth();
    lemma_alphabet_.StopGrowth();
    prefix_alphabet_.StopGrowth();
    suffix_alphabet_.StopGrowth();
    pos_alphabet_.StopGrowth();
    cpos_alphabet_.StopGrowth();
    LOG(INFO) << "Number of forms: " << form_alphabet_.size() << endl
              << "Number of lower-case forms: " << form_lower_alphabet_.size() << endl
              << "Number of lemmas: " << lemma_alphabet_.size() << endl
              << "Number of prefixes: " << prefix_alphabet_.size() << endl
              << "Number of suffixes: " << suffix_alphabet_.size() << endl
              << "Number of pos: " << pos_alphabet_.size() << endl
              << "Number of cpos: " << cpos_alphabet_.size();

    CHECK_LT(form_alphabet_.size(), 0xfffff);
    CHECK_LT(form_lower_alphabet_.size(), 0xfffff);
    CHECK_LT(lemma_alphabet_.size(), 0xfffff);
    CHECK_LT(prefix_alphabet_.size(), 0xffff);
    CHECK_LT(suffix_alphabet_.size(), 0xffff);
    CHECK_LT(pos_alphabet_.size(), 0xff);
    CHECK_LT(cpos_alphabet_.size(), 0xff);

    // TODO: Remove this (only for debugging purposes).
    BuildNames();
}
