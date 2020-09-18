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

#include "SemanticInstance.h"

void SemanticInstance::Initialize(const string &name,
                                  const vector<string> &forms,
                                  const vector<string> &lemmas,
                                  const vector<string> &cpos,
                                  const vector<string> &pos,
                                  const vector<Predicate> &predicates,
                                  const vector<vector<Argument> > &arguments) {
    name_ = name;
    forms_ = forms;
    lemmas_ = lemmas;
    cpostags_ = cpos;
    postags_ = pos;
    predicates_ = predicates;
    arguments_ = arguments;
}
