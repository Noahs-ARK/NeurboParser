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

#include "SemanticPipe.h"

#ifndef _WIN32

#else
#include <time.h>
#endif


using namespace std;


// Define the current model version and the oldest back-compatible version.
// The format is AAAA.BBBB.CCCC, e.g., 2 0003 0000 means "2.3.0".
const uint64_t kSemanticParserModelVersion = 200030000;
const uint64_t kOldestCompatibleSemanticParserModelVersion = 200030000;
const uint64_t kSemanticParserModelCheck = 1234567890;

DEFINE_bool(use_only_labeled_arc_features, true,
            "True for not using unlabeled arc features in addition to labeled ones.");
DEFINE_bool(use_only_labeled_sibling_features, false, //true,
            "True for not using unlabeled sibling features in addition to labeled ones.");
DEFINE_bool(use_labeled_sibling_features, false, //true,
            "True for using labels in sibling features.");

void SemanticPipe::SaveModel(FILE *fs) {
    bool success;
    success = WriteUINT64(fs, kSemanticParserModelCheck);
    CHECK(success);
    success = WriteUINT64(fs, kSemanticParserModelVersion);
    CHECK(success);
    token_dictionary_->Save(fs);
    dependency_dictionary_->Save(fs);
    Pipe::SaveModel(fs);
    return;
}

void SemanticPipe::SaveNeuralModel() {
    string file_path = options_->GetModelFilePath() + ".dynet";
    save_dynet_model(file_path, model_);
}

void SemanticPipe::SavePruner() {
    SemanticOptions *semantic_options = GetSemanticOptions();
    const string file_path
            = semantic_options->GetPrunerModelFilePath();
    save_dynet_model(file_path, pruner_model_);
}

void SemanticPipe::LoadModel(FILE *fs) {
    bool success;
    uint64_t model_check;
    uint64_t model_version;
    success = ReadUINT64(fs, &model_check);
    CHECK(success);
    CHECK_EQ(model_check, kSemanticParserModelCheck)
        << "The model file is too old and not supported anymore.";
    success = ReadUINT64(fs, &model_version);
    CHECK(success);
    CHECK_GE(model_version, kOldestCompatibleSemanticParserModelVersion)
        << "The model file is too old and not supported anymore.";
    delete token_dictionary_;
    CreateTokenDictionary();
    static_cast<SemanticDictionary *>(dictionary_)->
            SetTokenDictionary(token_dictionary_);
    token_dictionary_->Load(fs);
    CreateDependencyDictionary();
    dependency_dictionary_->SetTokenDictionary(token_dictionary_);
    static_cast<SemanticDictionary *>(dictionary_)->
            SetDependencyDictionary(dependency_dictionary_);
    dependency_dictionary_->Load(fs);
    options_->Load(fs);
    dictionary_->Load(fs);
}

void SemanticPipe::LoadNeuralModel() {
    if (model_) delete model_;
    if (parser_) delete parser_;
    if (trainer_) delete trainer_;
    model_ = new ParameterCollection();
    SemanticOptions *semantic_options = GetSemanticOptions();

    int num_roles = GetSemanticDictionary()->GetNumRoles();
    if (semantic_options->trainer() == "adadelta")
        trainer_ = new AdadeltaTrainer(*model_);
    else if (semantic_options->trainer() == "adam") {
        trainer_ = new AdamTrainer(*model_, 0.001, 0.9, 0.9, 1e-8);
    } else if (semantic_options->trainer() == "sgd") {
        trainer_ = new SimpleSGDTrainer(*model_);
    } else {
        LOG(INFO) << "Unsupported trainer. Giving up..." << endl;
        CHECK(false);
    }
    trainer_->clip_threshold = 1.0;
    parser_ = new Parser(semantic_options, num_roles, decoder_, model_);
    parser_->InitParams(model_);

    string file_path = options_->GetModelFilePath() + ".dynet";
    load_dynet_model(file_path, model_);
    // temporary solution to weight_decay issue in dynet
    // TODO: save the weight_decay along with the model.
//    model_->get_weight_decay().update_weight_decay(semantic_options->num_updates_);
    for (int i = 0;i < semantic_options->num_updates_; ++ i) {
        model_->get_weight_decay().update_weight_decay();
        if (model_->get_weight_decay().parameters_need_rescaled())
            model_->get_weight_decay().reset_weight_decay();
    }
//    LOG(INFO) <<model_->get_weight_decay().current_weight_decay();
}

void SemanticPipe::LoadPruner() {
    SemanticOptions *semantic_options = GetSemanticOptions();
    const string file_path = semantic_options->GetPrunerModelFilePath();
    if (pruner_model_) delete pruner_model_;
    if (pruner_) delete pruner_;
    if (pruner_trainer_) delete pruner_trainer_;

    pruner_model_ = new ParameterCollection();
    int num_roles = GetSemanticDictionary()->GetNumRoles();
    if (semantic_options->trainer() == "adadelta")
        pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
    else if (semantic_options->trainer() == "adam") {
        pruner_trainer_ = new AdamTrainer(*pruner_model_, 0.001, 0.9, 0.9, 1e-8);
    } else if (semantic_options->trainer() == "sgd") {
        pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_);
    } else {
        CHECK(false) << "Unsupported trainer. Giving up..." << endl;
    }
    pruner_trainer_->clip_threshold = 1.0;
    pruner_ = new Pruner(semantic_options, num_roles, decoder_, pruner_model_);
    pruner_->InitParams(pruner_model_);
    load_dynet_model(file_path, pruner_model_);
    // temporary solution to weight_decay issue in dynet
    // TODO: save the weight_decay along with the model.
//    pruner_model_->get_weight_decay().update_weight_decay(semantic_options->pruner_num_updates_);
    for (int i = 0;i < semantic_options->pruner_num_updates_; ++ i) {
        pruner_model_->get_weight_decay().update_weight_decay();
        if (pruner_model_->get_weight_decay().parameters_need_rescaled())
            pruner_model_->get_weight_decay().reset_weight_decay();
    }
}

void SemanticPipe::PreprocessData() {
    delete token_dictionary_;
    CreateTokenDictionary();
    static_cast<SemanticDictionary *>(dictionary_)->SetTokenDictionary(token_dictionary_);
    static_cast<DependencyTokenDictionary *>(token_dictionary_)->Initialize(GetSemanticReader());
    delete dependency_dictionary_;
    CreateDependencyDictionary();
    dependency_dictionary_->SetTokenDictionary(token_dictionary_);
    static_cast<SemanticDictionary *>(dictionary_)->SetDependencyDictionary(dependency_dictionary_);
    dependency_dictionary_->CreateLabelDictionary(GetSemanticReader());
    static_cast<SemanticDictionary *>(dictionary_)->CreatePredicateRoleDictionaries(GetSemanticReader());
}

void SemanticPipe::MakeParts(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    int slen =
            static_cast<SemanticInstanceNumeric *>(instance)->size();
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    semantic_parts->Initialize();
    bool make_gold = (gold_outputs != NULL);
    if (make_gold) gold_outputs->clear();

    if (GetSemanticOptions()->train_pruner()) {
        // For the pruner, make only unlabeled arc-factored and predicate parts and
        // compute indices.
        MakePartsBasic(instance, false, parts, gold_outputs);
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(slen, false);
    } else {
        // Make arc-factored and predicate parts and compute indices.
        MakePartsBasic(instance, parts, gold_outputs);
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(slen, GetSemanticOptions()->labeled());
    }
}

void SemanticPipe::MakePartsBasic(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    int slen =
            static_cast<SemanticInstanceNumeric *>(instance)->size();
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);

    MakePartsBasic(instance, false, parts, gold_outputs);
    semantic_parts->BuildOffsets();
    semantic_parts->BuildIndices(slen, false);

    // Prune using a basic first-order model.
    if (GetSemanticOptions()->prune_basic()) {
        Prune(instance, parts, gold_outputs, options_->train());
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(slen, false);
    }

    if (GetSemanticOptions()->labeled()) {
        MakePartsBasic(instance, true, parts, gold_outputs);
    }
}

void SemanticPipe::MakePartsBasic(Instance *instance, bool add_labeled_parts, Parts *parts,
                                  vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    int slen = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    bool prune_labels = semantic_options->prune_labels();
    bool prune_labels_with_relation_paths =
            semantic_options->prune_labels_with_relation_paths();
    bool prune_labels_with_senses = semantic_options->prune_labels_with_senses();
    bool prune_distances = semantic_options->prune_distances();
    bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();
    vector<int> allowed_labels;

    if (add_labeled_parts && !prune_labels) {
        allowed_labels.resize(semantic_dictionary->GetRoleAlphabet().size());
        for (int i = 0; i < allowed_labels.size(); ++i) {
            allowed_labels[i] = i;
        }
    }

    // Add predicate parts.
    int num_parts_initial = semantic_parts->size();
    if (!add_labeled_parts) {
        for (int p = 0; p < slen; ++p) {
            if (p == 0 && !allow_root_predicate) continue;
            int lemma_id = TOKEN_UNKNOWN;
            if (use_predicate_senses) {
                lemma_id = sentence->GetLemmaId(p);
                CHECK_GE(lemma_id, 0);
            }
            const vector<SemanticPredicate *> *predicates =
                    &semantic_dictionary->GetLemmaPredicates(lemma_id);
            if (predicates->size() == 0 && allow_unseen_predicates) {
                predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
            }
            for (int s = 0; s < predicates->size(); ++s) {
                Part *part = semantic_parts->CreatePartPredicate(p, s);
                semantic_parts->AddPart(part);
                if (make_gold) {
                    bool is_gold = false;
                    int k = sentence->FindPredicate(p);
                    if (k >= 0) {
                        int predicate_id = sentence->GetPredicateId(k);
                        if (!use_predicate_senses) {
                            CHECK_EQ((*predicates)[s]->id(), PREDICATE_UNKNOWN);
                        }
                        if (predicate_id < 0 || (*predicates)[s]->id() == predicate_id) {
                            is_gold = true;
                        }
                    }
                    if (is_gold) {
                        gold_outputs->push_back(1.0);
                    } else {
                        gold_outputs->push_back(0.0);
                    }
                }
            }
        }

        // Compute offsets for predicate parts.
        semantic_parts->SetOffsetPredicate(num_parts_initial, semantic_parts->size() - num_parts_initial);
    }

    // Add unlabeled/labeled arc parts.
    num_parts_initial = semantic_parts->size();
    for (int p = 0; p < slen; ++p) {
        if (p == 0 && !allow_root_predicate) continue;
        int lemma_id = TOKEN_UNKNOWN;
        if (use_predicate_senses) {
            lemma_id = sentence->GetLemmaId(p);
            CHECK_GE(lemma_id, 0);
        }
        const vector<SemanticPredicate *> *predicates =
                &semantic_dictionary->GetLemmaPredicates(lemma_id);
        if (predicates->size() == 0 && allow_unseen_predicates) {
            predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
        }
        for (int a = 1; a < slen; ++a) {
            if (!allow_self_loops && p == a) continue;
            for (int s = 0; s < predicates->size(); ++s) {
                int arc_index = -1;
                if (add_labeled_parts) {
                    // If no unlabeled arc is there, just skip it.
                    // This happens if that arc was pruned out.
                    arc_index = semantic_parts->FindArc(p, a, s);
                    if (0 > arc_index) {
                        continue;
                    }
                } else {
                    if (prune_distances) {
                        int predicate_pos_id = sentence->GetPosId(p);
                        int argument_pos_id = sentence->GetPosId(a);
                        if (p < a) {
                            // Right attachment.
                            if (a - p > semantic_dictionary->GetMaximumRightDistance
                                    (predicate_pos_id, argument_pos_id))
                                continue;
                        } else {
                            // Left attachment.
                            if (p - a > semantic_dictionary->GetMaximumLeftDistance
                                    (predicate_pos_id, argument_pos_id))
                                continue;
                        }
                    }
                }

                if (prune_labels_with_relation_paths) {
                    int relation_path_id = sentence->GetRelationPathId(p, a);
                    allowed_labels.clear();
                    if (relation_path_id >= 0 &&
                        relation_path_id < semantic_dictionary->
                                GetRelationPathAlphabet().size()) {
                        allowed_labels = semantic_dictionary->
                                GetExistingRolesWithRelationPath(relation_path_id);
                        //LOG(INFO) << "Path: " << relation_path_id << " Roles: " << allowed_labels.size();
                    }
                    set<int> label_set;
                    for (int m = 0; m < allowed_labels.size(); ++m) {
                        if (!prune_labels_with_senses ||
                            (*predicates)[s]->HasRole(allowed_labels[m])) {
                            label_set.insert(allowed_labels[m]);
                        }
                    }
                    allowed_labels.clear();
                    for (set<int>::iterator it = label_set.begin();
                         it != label_set.end(); ++it) {
                        allowed_labels.push_back(*it);
                    }
                    if (!add_labeled_parts && allowed_labels.empty()) {
                        continue;
                    }
                } else if (prune_labels) {
                    // TODO: allow both kinds of label pruning simultaneously?
                    int predicate_pos_id = sentence->GetPosId(p);
                    int argument_pos_id = sentence->GetPosId(a);
                    allowed_labels.clear();
                    allowed_labels = semantic_dictionary->
                            GetExistingRoles(predicate_pos_id, argument_pos_id);
                    set<int> label_set;
                    for (int m = 0; m < allowed_labels.size(); ++m) {
                        if (!prune_labels_with_senses ||
                            (*predicates)[s]->HasRole(allowed_labels[m])) {
                            label_set.insert(allowed_labels[m]);
                        }
                    }
                    allowed_labels.clear();
                    for (set<int>::iterator it = label_set.begin();
                         it != label_set.end(); ++it) {
                        allowed_labels.push_back(*it);
                    }
                    if (!add_labeled_parts && allowed_labels.empty()) {
                        continue;
                    }
                }

                // Add parts for labeled/unlabeled arcs.
                if (add_labeled_parts) {
                    // If there is no allowed label for this arc, but the unlabeled arc was added,
                    // then it was forced to be present for some reason (e.g. to maintain connectivity of the
                    // graph). In that case (which should be pretty rare) consider all the
                    // possible labels.
                    if (allowed_labels.empty()) {
                        allowed_labels.resize(semantic_dictionary->GetRoleAlphabet().size());
                        for (int role = 0; role < allowed_labels.size(); ++role) {
                            allowed_labels[role] = role;
                        }
                    }

                    for (int m = 0; m < allowed_labels.size(); ++m) {
                        int role = allowed_labels[m];
                        if (prune_labels && prune_labels_with_senses) {
                            CHECK((*predicates)[s]->HasRole(role));
                        }

                        Part *part = semantic_parts->CreatePartLabeledArc(p, a, s, role);
                        CHECK_GE(arc_index, 0);
                        semantic_parts->AddLabeledPart(part, arc_index);
                        if (make_gold) {
                            int k = sentence->FindPredicate(p);
                            int l = sentence->FindArc(p, a);
                            bool is_gold = false;

                            if (k >= 0 && l >= 0) {
                                int predicate_id = sentence->GetPredicateId(k);
                                int argument_id = sentence->GetArgumentRoleId(k, l);
                                if (!use_predicate_senses) {
                                    CHECK_EQ((*predicates)[s]->id(), PREDICATE_UNKNOWN);
                                }
                                //if (use_predicate_senses) CHECK_LT(predicate_id, 0);
                                if ((predicate_id < 0 ||
                                     (*predicates)[s]->id() == predicate_id) &&
                                    role == argument_id) {
                                    is_gold = true;
                                }
                            }
                            if (is_gold) {
                                gold_outputs->push_back(1.0);
                            } else {
                                gold_outputs->push_back(0.0);
                            }
                        }
                    }
                } else {
                    Part *part = semantic_parts->CreatePartArc(p, a, s);
                    semantic_parts->AddPart(part);
                    if (make_gold) {
                        int k = sentence->FindPredicate(p);
                        int l = sentence->FindArc(p, a);
                        bool is_gold = false;
                        if (k >= 0 && l >= 0) {
                            int predicate_id = sentence->GetPredicateId(k);
                            if (!use_predicate_senses) {
                                CHECK_EQ((*predicates)[s]->id(), PREDICATE_UNKNOWN);
                            }
                            if (predicate_id < 0 || (*predicates)[s]->id() == predicate_id) {
                                is_gold = true;
                            }
                        }
                        if (is_gold) {
                            gold_outputs->push_back(1.0);
                        } else {
                            gold_outputs->push_back(0.0);
                        }
                    }
                }
            }
        }
    }

    // Compute offsets for labeled/unlabeled arcs.
    if (!add_labeled_parts) {
        semantic_parts->SetOffsetArc(num_parts_initial, semantic_parts->size() - num_parts_initial);
    } else {
        semantic_parts->SetOffsetLabeledArc(num_parts_initial, semantic_parts->size() - num_parts_initial);
    }
}

void SemanticPipe::Prune(Instance *instance, Parts *parts,
                         vector<double> *gold_outputs,
                         bool preserve_gold) {
    SemanticParts *semantic_parts
            = static_cast<SemanticParts *>(parts);
    vector<double> scores;
    vector<double> predicted_outputs;

    // Make sure gold parts are only preserved at training time.
    CHECK(!preserve_gold || options_->train());
    if (!gold_outputs) preserve_gold = false;

    ComputationGraph cg;
    Expression ex_loss
            = pruner_->BuildGraph(instance, parts, scores,
                                  *gold_outputs, predicted_outputs,
                                  false, 0.0,
                                  form_count_,
                                  false, cg);
    double loss = as_scalar(cg.forward(ex_loss));

    int offset_predicate_parts, num_predicate_parts;
    int offset_arcs, num_arcs;
    semantic_parts->GetOffsetPredicate(&offset_predicate_parts,
                                       &num_predicate_parts);
    semantic_parts->GetOffsetArc(&offset_arcs, &num_arcs);

    double threshold = 0.5;
    int r0 = offset_arcs; // Preserve all the predicate parts.
    semantic_parts->ClearOffsets();
    semantic_parts->SetOffsetPredicate(offset_predicate_parts,
                                       num_predicate_parts);
    for (int r = 0; r < num_arcs; ++r) {
        // Preserve gold parts (at training time).
        if (predicted_outputs[offset_arcs + r] >= threshold ||
            (preserve_gold && (*gold_outputs)[offset_arcs + r] >= threshold)) {
            (*parts)[r0] = (*parts)[offset_arcs + r];
            semantic_parts->
                    SetLabeledParts(r0, semantic_parts->GetLabeledParts(offset_arcs + r));
            if (gold_outputs) {
                (*gold_outputs)[r0] = (*gold_outputs)[offset_arcs + r];
            }
            ++r0;
        } else {
            delete (*parts)[offset_arcs + r];
        }
    }

    if (gold_outputs) gold_outputs->resize(r0);
    semantic_parts->Resize(r0);
    semantic_parts->DeleteIndices();
    semantic_parts->SetOffsetArc(offset_arcs,
                                 parts->size() - offset_arcs);
}

void SemanticPipe::LabelInstance(Parts *parts, const vector<double> &output, Instance *instance) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticInstance *semantic_instance =
            static_cast<SemanticInstance *>(instance);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();

    int instance_length = semantic_instance->size();
    double threshold = 0.5;
    semantic_instance->ClearPredicates();
    for (int p = 0; p < instance_length; ++p) {
        //if (p == 0 && !allow_root_predicate) continue;
        const vector<int> &senses = semantic_parts->GetSenses(p);
        vector<int> argument_indices;
        vector<string> argument_roles;
        int predicted_sense = -1;
        for (int k = 0; k < senses.size(); k++) {
            int s = senses[k];
            for (int a = 1; a < instance_length; ++a) {
                if (GetSemanticOptions()->labeled()) {
                    int r = semantic_parts->FindArc(p, a, s);
                    if (r < 0) continue;
                    const vector<int> &labeled_arcs =
                            semantic_parts->FindLabeledArcs(p, a, s);
                    for (int l = 0; l < labeled_arcs.size(); ++l) {
                        int r = labeled_arcs[l];
                        CHECK_GE(r, 0);
                        CHECK_LT(r, parts->size());
                        if (output[r] > threshold) {
                            if (predicted_sense != s) {
                                CHECK_LT(predicted_sense, 0);
                                predicted_sense = s;
                            }
                            argument_indices.push_back(a);
                            SemanticPartLabeledArc *labeled_arc =
                                    static_cast<SemanticPartLabeledArc *>((*parts)[r]);
                            string role =
                                    semantic_dictionary->GetRoleName(labeled_arc->role());
                            argument_roles.push_back(role);
                        }
                    }
                } else {
                    int r = semantic_parts->FindArc(p, a, s);
                    if (r < 0) continue;
                    if (output[r] > threshold) {
                        if (predicted_sense != s) {
                            CHECK_LT(predicted_sense, 0);
                            predicted_sense = s;
                        }
                        argument_indices.push_back(a);
                        argument_roles.push_back("ARG");
                    }
                }
            }
        }

        if (predicted_sense >= 0) {
            int s = predicted_sense;
            // Get the predicate id for this part.
            // TODO(atm): store this somewhere, so that we don't need to recompute this
            // all the time. Maybe store this directly in arc->sense()?
            int lemma_id = TOKEN_UNKNOWN;
            if (GetSemanticOptions()->use_predicate_senses()) {
                lemma_id = semantic_dictionary->GetTokenDictionary()->
                        GetLemmaId(semantic_instance->GetLemma(p));
                if (lemma_id < 0) lemma_id = TOKEN_UNKNOWN;
            }
            const vector<SemanticPredicate *> *predicates =
                    &GetSemanticDictionary()->GetLemmaPredicates(lemma_id);
            if (predicates->size() == 0 &&
                GetSemanticOptions()->allow_unseen_predicates()) {
                predicates = &GetSemanticDictionary()->GetLemmaPredicates(TOKEN_UNKNOWN);
            }
            int predicate_id = (*predicates)[s]->id();
            string predicate_name =
                    semantic_dictionary->GetPredicateName(predicate_id);
            semantic_instance->AddPredicate(predicate_name, p, argument_roles,
                                            argument_indices);
        }
    }
}

void SemanticPipe::Train() {
    CreateInstances();
    SemanticOptions *semantic_options = GetSemanticOptions();
    LoadPruner();
    if (semantic_options->use_pretrained_embedding()) {
        LoadPretrainedEmbedding(true, false);
    }
    BuildFormCount();
    vector<int> idxs;
    for (int i = 0; i < instances_.size(); ++i) {
        idxs.push_back(i);
    }

    double unlabeled_F1 = 0, labeled_F1 = 0, best_labeled_F1 = -1;

    for (int i = 0; i < options_->GetNumEpochs(); ++i) {
        semantic_options->train_on();
        random_shuffle(idxs.begin(), idxs.end());
        TrainEpoch(idxs, i);
        semantic_options->train_off();
        Run(unlabeled_F1, labeled_F1);
        if (labeled_F1 > best_labeled_F1 && labeled_F1 > 0.60) {
            SaveNeuralModel();
            SaveModelFile();
            best_labeled_F1 = labeled_F1;
        }
    }
}

void SemanticPipe::TrainPruner() {
    CreateInstances();
    SemanticOptions *semantic_options = GetSemanticOptions();
    if (semantic_options->use_pretrained_embedding()) {
        LoadPretrainedEmbedding(false, true);
    }
    BuildFormCount();

    vector<int> idxs;
    for (int i = 0; i < instances_.size(); ++i) {
        idxs.push_back(i);
    }
    for (int i = 0; i < semantic_options->pruner_epochs(); ++i) {
        semantic_options->train_on();
        random_shuffle(idxs.begin(), idxs.end());
        TrainPrunerEpoch(idxs, i);
        semantic_options->train_off();
        SavePruner();
        SaveModelFile();
    }
}

double SemanticPipe::TrainEpoch(const vector<int> &idxs, int epoch) {
    Instance *instance;
    Parts *parts = CreateParts();
    vector<double> scores;
    vector<double> gold_outputs;
    vector<double> predicted_outputs;

    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_word_dropout = semantic_options->use_word_dropout();
    float word_dropout_rate = semantic_options->word_dropout_rate();
    if (use_word_dropout) {
        CHECK(word_dropout_rate > 0.0 && word_dropout_rate < 1.0);
    }
    float loss = 0.0;
    int num_instances = idxs.size();
    if (epoch == 0) {
        LOG(INFO) << "Number of instances: " << num_instances << endl;
    }
    LOG(INFO) << " Iteration #" << epoch + 1;
    for (int i = 0; i < num_instances; i++) {
        instance = instances_[idxs[i]];
        MakeParts(instance, parts, &gold_outputs);
        ComputationGraph cg;
        Expression ex_loss = parser_->BuildGraph(instance, parts, scores,
                                                 gold_outputs, predicted_outputs,
                                                 use_word_dropout, word_dropout_rate,
                                                 form_count_,
                                                 true, cg);
        loss += max(float(0.0), as_scalar(cg.forward(ex_loss)));
        int corr = 0;
        for (int r = 0; r < parts->size(); ++r) {
            if (NEARLY_EQ_TOL(gold_outputs[r], predicted_outputs[r], 1e-6))
                corr += 1;
            else
                break;
        }
        if (corr < parts->size()) {
            cg.backward(ex_loss);
            trainer_->update();
            ++semantic_options->num_updates_;
        }
    }
    delete parts;
    LOG(INFO) << "Training loss: " << loss / num_instances << endl;
    return loss;
}

double SemanticPipe::TrainPrunerEpoch(const vector<int> &idxs, int epoch) {
    Instance *instance;
    Parts *parts = CreateParts();
    vector<double> scores, gold_outputs, predicted_outputs;
    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_word_dropout = semantic_options->use_word_dropout();
    float word_dropout_rate = semantic_options->word_dropout_rate();
    if (use_word_dropout) {
        CHECK(word_dropout_rate > 0.0 && word_dropout_rate < 1.0);
    }
    double loss = 0.0;
    int num_instances = idxs.size();
    if (epoch == 0) {
        LOG(INFO) << "Number of instances: " << num_instances << endl;
    }
    LOG(INFO) << " Iteration #" << epoch + 1;
    for (int i = 0; i < num_instances; i++) {
        instance = instances_[idxs[i]];
        MakeParts(instance, parts, &gold_outputs);
        ComputationGraph cg;
        Expression ex_loss = pruner_->BuildGraph(instance, parts, scores,
                                                 gold_outputs, predicted_outputs,
                                                 use_word_dropout, word_dropout_rate,
                                                 form_count_,
                                                 true, cg);
        loss += as_scalar(cg.forward(ex_loss));
        cg.backward(ex_loss);
        pruner_trainer_->update();
        ++semantic_options->pruner_num_updates_;
    }
    delete parts;
    LOG(INFO) << "training loss: " << loss / num_instances << endl;
    return loss;
}

void SemanticPipe::Test() {
    SemanticOptions *semantic_options = GetSemanticOptions();
    LoadNeuralModel();
    LoadPruner();
    semantic_options->train_off();
    double unlabeled_F1 = 0, labeled_F1 = 0;
    Run(unlabeled_F1, labeled_F1);
}

void SemanticPipe::Run(double &unlabeled_F1, double &labeled_F1) {
    Parts *parts = CreateParts();
    vector<double> scores;
    vector<double> gold_outputs;
    vector<double> predicted_outputs;

    timeval start, end;
    gettimeofday(&start, NULL);

    if (options_->evaluate()) BeginEvaluation();
    reader_->Open(options_->GetTestFilePath());
    writer_->Open(options_->GetOutputFilePath());
    int num_instances = 0;
    Instance *instance = reader_->GetNext();
    double forward_loss = 0.0;
    while (instance) {
        Instance *formatted_instance = GetFormattedInstance(instance);
        MakeParts(formatted_instance, parts, &gold_outputs);
        SemanticParts *sp = static_cast<SemanticParts *> (parts);
        int offset, num;
        sp->GetOffsetArc(&offset, &num);
        ComputationGraph cg;
        Expression ex_loss = parser_->BuildGraph(formatted_instance, parts, scores,
                                                 gold_outputs, predicted_outputs,
                                                 false, 0.0,
                                                 form_count_,
                                                 false, cg);
        double loss = as_scalar(cg.forward(ex_loss));
        loss = max(loss, 0.0);
        forward_loss += loss;
        Instance *output_instance = instance->Copy();
        LabelInstance(parts, predicted_outputs, output_instance);
        if (options_->evaluate()) {
            EvaluateInstance(instance, output_instance,
                             parts, gold_outputs, predicted_outputs);
        }
        writer_->Write(output_instance);
        if (formatted_instance != instance) delete formatted_instance;
        delete output_instance;
        delete instance;

        instance = reader_->GetNext();
        ++num_instances;
    }
    forward_loss /= num_instances;
    LOG(INFO) << "dev loss: " << forward_loss << endl;
    delete parts;
    writer_->Close();
    reader_->Close();

    gettimeofday(&end, NULL);

    if (options_->evaluate()) EndEvaluation(unlabeled_F1, labeled_F1);
}

void SemanticPipe::LoadPretrainedEmbedding(bool load_parser_embedding, bool load_pruner_embedding) {
    SemanticOptions *semantic_option = GetSemanticOptions();
    embedding_ = new unordered_map<int, vector<float>>();
    unsigned dim = semantic_option->word_dim();
    ifstream in(semantic_option->GetPretrainedEmbeddingFilePath());

    CHECK(in.is_open()) << "Pretrained embeddings FILE NOT FOUND!" << endl;

    string line;
    getline(in, line);
    vector<float> v(dim, 0);
    string word;
    int found = 0;
    while (getline(in, line)) {
        istringstream lin(line);
        lin >> word;
        for (unsigned i = 0; i < dim; ++i)
            lin >> v[i];
        int form_id = token_dictionary_->GetFormId(word);
        if (form_id < 0)
            continue;
        found += 1;
        (*embedding_)[form_id] = v;
    }
    in.close();
    LOG(INFO) << found << "/" << token_dictionary_->GetNumForms() << " words found in the pretrained embedding" << endl;
    if (load_parser_embedding) parser_->LoadEmbedding(embedding_);
    if (load_pruner_embedding) pruner_->LoadEmbedding(embedding_);
    delete embedding_;
}

void SemanticPipe::BuildFormCount() {
    form_count_ = new unordered_map<int, int>();
    for (int i = 0; i < instances_.size(); i++) {
        Instance *instance = instances_[i];
        SemanticInstanceNumeric *sentence =
                static_cast<SemanticInstanceNumeric *>(instance);
        const vector<int> form_ids = sentence->GetFormIds();
        for (int r = 0; r < form_ids.size(); ++r) {
            int form_id = form_ids[r];
            CHECK_NE(form_id, UNK_ID);
            if (form_count_->find(form_id) == form_count_->end()) {
                (*form_count_)[form_id] = 1;
            } else {
                (*form_count_)[form_id] += 1;
            }
        }
    }
}