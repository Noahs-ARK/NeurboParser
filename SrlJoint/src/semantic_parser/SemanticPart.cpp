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

#include "SemanticPart.h"

void SemanticParts::DeleteAll() {
    for (int i = 0; i < NUM_SEMANTICPARTS; ++i) {
        offsets_[i] = -1;
    }

    DeleteIndices();
    for (iterator iter = begin(); iter != end(); iter++) {
        if ((*iter) != NULL) {
            delete (*iter);
            *iter = NULL;
        }
    }
    clear();
}

void SemanticParts::DeleteIndices() {
    predicate_index_.clear();
    for (int p = 0;p < argument_index_.size(); ++ p) {
        argument_index_[p].clear();
    }
    argument_index_.clear();
}

void SemanticParts::BuildIndices() {
    DeleteIndices();
    int offset_predicate_parts, num_predicate_parts;
    GetOffsetPredicate(&offset_predicate_parts, &num_predicate_parts);
    int offset_unlabeled_argument_parts, num_unlabeled_argument_parts;
    GetOffsetUnlabeledArgument(&offset_unlabeled_argument_parts, &num_unlabeled_argument_parts);
    int offset_argument_parts, num_argument_parts;
    GetOffsetArgument(&offset_argument_parts, &num_argument_parts);
    for (int r = 0; r < num_predicate_parts; ++ r) {
        Part *part = (*this)[offset_predicate_parts + r];
        SemanticPartPredicate *predicate = static_cast<SemanticPartPredicate *> (part);
        predicate_index_[*predicate] = offset_predicate_parts + r;
    }
    argument_index_.clear();
    int s, e;
    for (int r = 0; r < num_unlabeled_argument_parts; ++ r) {
        Part *part = (*this)[offset_unlabeled_argument_parts + r];
        SemanticPartArgument *argument = static_cast<SemanticPartArgument*>(part);
        int p = argument->frame();
        argument->span(s, e);
        CHECK(argument_index_[p].find(*argument) == argument_index_[p].end());
        argument_index_[p][*argument] = offset_unlabeled_argument_parts + r;
    }
    for (int r = 0; r < num_argument_parts; ++ r) {
        Part *part = (*this)[offset_argument_parts + r];
        SemanticPartArgument *argument = static_cast<SemanticPartArgument*>(part);
        CHECK_GE(argument->role(), 0);
        int p = argument->frame();
        argument->span(s, e);
        CHECK(argument_index_[p].find(*argument) == argument_index_[p].end());
        argument_index_[p][*argument] = offset_argument_parts + r;
    }
}
