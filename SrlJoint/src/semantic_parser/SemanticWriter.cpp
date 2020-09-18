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

#include <glog/logging.h>
#include "SemanticWriter.h"
#include "SemanticInstance.h"
#include "SemanticOptions.h"

//5	critics	_	critic	nns	NNS	1	_	_	_	_	_	critic.n	Judgment_communication	S-Communicator
void SemanticWriter::Write(Instance *instance) {
    SemanticInstance *semantic_instance =
            static_cast<SemanticInstance *>(instance);
    if (semantic_instance->GetName() != "") {
        os_ << semantic_instance->GetName() << endl;
    }
    vector<int> top_nodes;
	int num_pred = semantic_instance->GetNumPredicates();
	CHECK_LE(num_pred, 1);
    for (int i = 1; i < semantic_instance->size() - 1; ++i) {
        os_ << i << "\t";
        os_ << semantic_instance->GetForm(i) << "\t";
	    os_ << "_\t";
        os_ << semantic_instance->GetLemma(i) << "\t";
        os_ << semantic_instance->GetPosTag(i) << "\t";
	    os_ << semantic_instance->GetPosTag(i) << "\t";
	    os_ << semantic_instance->GetPosTag(i) << "\t";
	    os_ << "id\t";
	    os_ << "_\t_\t_\t_\t_\t";

	    bool is_pred = false;
	    for (int j = 0;j < num_pred; ++ j) {
		    if (is_pred) break;
		    int s, e;
		    semantic_instance->GetPredicateIndex(j, s, e);
		    if (s <= i && i <= e) {
			    is_pred = true;
		    }
	    }
	    if (is_pred) {
		    os_ << semantic_instance->GetPredicateName(0) << "\t";
		    os_ << semantic_instance->GetPredicateFrame(0) << "\t";
	    } else {
		    os_ << "_\t_\t";
	    }

	    int num_args = semantic_instance->GetNumArgument(0);
	    int arg_j = -1;
	    for (int j = 0;j < num_args; ++ j) {
		    if (arg_j >= 0) break;
		    int s, e;
		    semantic_instance->GetArgumentIndex(0, j, s, e);
		    if (s <= i && i <= e) {
			    arg_j = j;
		    }
	    }
	    if (arg_j >= 0) {
		    os_ << semantic_instance->GetArgumentRole(0, arg_j);
	    } else os_ << "_";
        os_ << endl;
    }
    os_ << endl;
}
