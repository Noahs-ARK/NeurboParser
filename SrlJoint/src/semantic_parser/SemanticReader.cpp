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

#include <algorithm>
#include "SemanticReader.h"
#include "SemanticOptions.h"
#include "Utils.h"

using namespace std;

Instance *SemanticReader::GetNext() {
    SemanticOptions *semantic_options =
            static_cast<SemanticOptions *>(options_);
    const string &format = semantic_options->file_format();

    // Fill all fields for the entire sentence.
    string name = "";
    vector<vector<vector<string>>> instance_fields(0);
    vector<vector<string>> frame_fields(0);
    string line;
    if (is_.is_open()) {
        while (!is_.eof()) {
            getline(is_, line);
            if (0 == line.compare("# end_of_instance")) {
                instance_fields.push_back(frame_fields);
                frame_fields.clear();
                getline(is_, line);
                break;
            }
            if (0 == line.substr(0, 13).compare("# instance_id")) {
                //LOG(INFO) << line;
                if (name != "") {
                    name += "\n" + line;
                } else {
                    name = line;
                }
                continue; // Sentence ID.
            }
            if (line.length() == 0) {
                if (frame_fields.size() > 1) {
                    instance_fields.push_back(frame_fields);
                    frame_fields.clear();
                }
                continue;
            }
            vector<string> fields;
            StringSplit(line, "\t", &fields, true);
            frame_fields.push_back(fields);
        }
    }
    bool read_next_sentence = false;
    if (!is_.eof()) read_next_sentence = true;
    if (!read_next_sentence) return NULL;

    // Sentence length.
    int num_frames = instance_fields.size();
    int length = instance_fields[0].size();
    CHECK(length > 0 && num_frames > 0);

    // Convert to array of forms, lemmas, etc.
    // Note: the first token is the start symbol, end the last the end
    vector<string> forms(length + 2);
    vector<string> lemmas(length + 2);
    vector<string> cpos(length + 2);
    vector<string> pos(length + 2);

    vector<Predicate> predicates(num_frames);
    vector<vector<Argument>> arguments(num_frames, vector<Argument>());
    // some frames may not have any argument

    forms[0] = kStart, forms[forms.size() - 1] = kEnd;
    lemmas[0] = kStart, lemmas[lemmas.size() - 1] = kEnd;
    cpos[0] = kStart, lemmas[lemmas.size() - 1] = kEnd;
    pos[0] = kStart, pos[pos.size() - 1] = kEnd;

    int p_start = -1, p_end = -1;
    int a_start = -1, a_end = -1;
    string lu_name, frame, argument_role;
    vector<string> bio;

    for (int f = 0; f < num_frames;++ f) {
        for (int i = 0; i < length; i++) {
            bio.clear();
            const vector<string> &info = instance_fields[f][i];
            forms[i + 1] = info[1];
            lemmas[i + 1] = info[3];
            cpos[i + 1] = info[5];
            pos[i + 1] = info[5];

            string pn = info[12];
            string pf = info[13];
            string BIO_tag = info[14];

            if (pn != "_") {
                // predicate span
                lu_name = pn;
                frame = pf;
                if (p_start != -1) {
                    p_end = i + 1;
                } else {
                    p_start = p_end = i + 1;
                }
            }

            if (BIO_tag != "_") {
                StringSplit(BIO_tag, "-", &bio, true);
	            if (bio.size() != 2) {
		            LOG(INFO) << BIO_tag;
	            }
                CHECK_EQ(2, bio.size());
                if (bio[0] == "B" && a_start == -1) {
                    a_start = i + 1;
                    argument_role = bio[1];
                } else if (bio[0] == "O") {
                    CHECK_EQ(bio[1], argument_role);
                    a_end = i + 1;
                } else if (bio[0] == "S") {
                    a_start = a_end = i + 1;
                    argument_role = bio[1];
                }
            }
            if (a_start >= 0 && a_end >= 0) {
                if (bio.size() != 2) {
                    LOG(INFO) << BIO_tag;
                }
                CHECK_EQ(bio.size(), 2);
                CHECK_LE(a_start, a_end);
                arguments[f].push_back(Argument(a_start, a_end, argument_role));
                a_start = a_end = -1;
            }
        }
        CHECK_LE(p_start, p_end);
        predicates[f] = Predicate(p_start, p_end, lu_name, frame);
        p_start = p_end = -1;
    }

    CHECK_EQ(arguments.size(), predicates.size());
    CHECK_EQ(predicates.size(), 1);
    SemanticInstance *instance = NULL;
    if (read_next_sentence && length >= 0) {
        instance = new SemanticInstance;
        instance->Initialize(name, forms, lemmas, cpos, pos,
                             predicates, arguments);
    }
    return static_cast<Instance *>(instance);
}
