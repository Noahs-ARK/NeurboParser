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

#include "NeuralSemanticPipe.h"
#include <queue>

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


void NeuralSemanticPipe::SaveModel(FILE *fs) {
    bool success;
    success = WriteUINT64(fs, kSemanticParserModelCheck);
    CHECK(success);
    success = WriteUINT64(fs, kSemanticParserModelVersion);
    CHECK(success);
    task1_token_dictionary_->Save(fs);
    task1_dependency_dictionary_->Save(fs);
    task1_dictionary_->Save(fs);
    task2_token_dictionary_->Save(fs);
    task2_dependency_dictionary_->Save(fs);
    task2_dictionary_->Save(fs);
    task3_token_dictionary_->Save(fs);
    task3_dependency_dictionary_->Save(fs);
    task3_dictionary_->Save(fs);
    options_->Save(fs);
//    parameters_->Save(fs);
    //Pipe::SaveModel(fs);
//    task1_pruner_parameters_->Save(fs);
//    task2_pruner_parameters_->Save(fs);
//    task3_pruner_parameters_->Save(fs);

    return;
}

void NeuralSemanticPipe::SaveNueralModel() {
    string dynet_model_path = options_->GetModelFilePath() + ".dynet";
    ofstream outfile(dynet_model_path);
    if (!outfile.is_open()) {
        LOG(INFO) << "Outfile opening failed: " << dynet_model_path << endl;
        return;
    }
    boost::archive::text_oarchive oa(outfile);
    oa & (*parser);
    oa & (*model);
    outfile.close();
}

void NeuralSemanticPipe::LoadModel(FILE *fs) {
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
    delete task1_token_dictionary_;
    delete task2_token_dictionary_;
    delete task3_token_dictionary_;
    CreateTokenDictionary();
    CreateDependencyDictionary();

    static_cast<SemanticDictionary *>(task1_dictionary_)->
            SetTokenDictionary(task1_token_dictionary_);
    task1_token_dictionary_->Load(fs);
    task1_dependency_dictionary_->SetTokenDictionary(task1_token_dictionary_);
    static_cast<SemanticDictionary *>(task1_dictionary_)->
            SetDependencyDictionary(task1_dependency_dictionary_);
    task1_dependency_dictionary_->Load(fs);
    task1_dictionary_->Load(fs);

    static_cast<SemanticDictionary *>(task2_dictionary_)->
            SetTokenDictionary(task2_token_dictionary_);
    task2_token_dictionary_->Load(fs);
    task2_dependency_dictionary_->SetTokenDictionary(task2_token_dictionary_);
    static_cast<SemanticDictionary *>(task2_dictionary_)->
            SetDependencyDictionary(task2_dependency_dictionary_);
    task2_dependency_dictionary_->Load(fs);
    task2_dictionary_->Load(fs);

    static_cast<SemanticDictionary *>(task3_dictionary_)->
            SetTokenDictionary(task3_token_dictionary_);
    task3_token_dictionary_->Load(fs);
    task3_dependency_dictionary_->SetTokenDictionary(task3_token_dictionary_);
    static_cast<SemanticDictionary *>(task3_dictionary_)->
            SetDependencyDictionary(task3_dependency_dictionary_);
    task3_dependency_dictionary_->Load(fs);
    task3_dictionary_->Load(fs);
    options_->Load(fs);
    return;

}

void NeuralSemanticPipe::LoadNueralModel() {
    model = new dynet::Model();
    SemanticOptions *semantic_options = GetSemanticOptions();
    int task1_num_roles = GetSemanticDictionary("task1")->GetNumRoles();
    int task2_num_roles = GetSemanticDictionary("task2")->GetNumRoles();
    int task3_num_roles = GetSemanticDictionary("task3")->GetNumRoles();
    if (semantic_options->output_term() == "shared1") {
        parser = new Shared1<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles,
                                                 model);
    } else if (semantic_options->output_term() == "shared3") {
        parser = new Shared3<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles,
                                                 model);
    } else if (semantic_options->output_term() == "freda1") {
        parser = new Freda1<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles,
                                                model);
    } else if (semantic_options->output_term() == "freda3") {
        parser = new Freda3<dynet::LSTMBuilder>(semantic_options, task1_num_roles, task2_num_roles, task3_num_roles,
                                                model);
    } else {
        CHECK(false) << "Unsupported output term. Giving up." << endl;
    }
    if (semantic_options->trainer() == "adadelta")
        trainer = new dynet::AdadeltaTrainer(*model);
    else if (semantic_options->trainer() == "adam") {
        trainer = new dynet::AdamTrainer(*model, 0.001, 0.9, 0.9, 1e-8);
    } else if (semantic_options->trainer() == "sgd") {
        trainer = new dynet::MomentumSGDTrainer(*model);
        trainer->eta_decay = 0.02;
    }
    trainer->clip_threshold = 1.0;
    string dynet_model_path = options_->GetModelFilePath() + ".dynet";
    ifstream infile(dynet_model_path);
    if (!infile.is_open()) {
        LOG(INFO) << "Infile opening failed: " << dynet_model_path << endl;
        return;
    }
    boost::archive::text_iarchive ia(infile);
    ia & (*parser);
    parser->InitParams(model);
    ia & (*model);
    infile.close();
    return;
}

void NeuralSemanticPipe::PreprocessData(const string &formalism) {
    if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
        LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        CHECK(1 == 0);
    }

    if (formalism == "task1") {
        static_cast<SemanticDictionary *>(task1_dictionary_)->SetTokenDictionary(task1_token_dictionary_);
        static_cast<DependencyTokenDictionary *>(task1_token_dictionary_)->Initialize(formalism,
                                                                                      GetSemanticReader(formalism));
        task1_dependency_dictionary_->SetTokenDictionary(task1_token_dictionary_);
        static_cast<SemanticDictionary *>(task1_dictionary_)->SetDependencyDictionary(task1_dependency_dictionary_);
        task1_dependency_dictionary_->CreateLabelDictionary(formalism, GetSemanticReader(formalism));
        static_cast<SemanticDictionary *>(task1_dictionary_)->CreatePredicateRoleDictionaries(formalism,
                                                                                              GetSemanticReader(
                                                                                                      formalism));
    } else if (formalism == "task2") {
        static_cast<SemanticDictionary *>(task2_dictionary_)->SetTokenDictionary(task2_token_dictionary_);
        static_cast<DependencyTokenDictionary *>(task2_token_dictionary_)->Initialize(formalism,
                                                                                      GetSemanticReader(formalism));
        task2_dependency_dictionary_->SetTokenDictionary(task2_token_dictionary_);
        static_cast<SemanticDictionary *>(task2_dictionary_)->SetDependencyDictionary(task2_dependency_dictionary_);
        task2_dependency_dictionary_->CreateLabelDictionary(formalism, GetSemanticReader(formalism));
        static_cast<SemanticDictionary *>(task2_dictionary_)->CreatePredicateRoleDictionaries(formalism,
                                                                                              GetSemanticReader(
                                                                                                      formalism));
    } else if (formalism == "task3") {
        static_cast<SemanticDictionary *>(task3_dictionary_)->SetTokenDictionary(task3_token_dictionary_);
        static_cast<DependencyTokenDictionary *>(task3_token_dictionary_)->Initialize(formalism,
                                                                                      GetSemanticReader(formalism));
        task3_dependency_dictionary_->SetTokenDictionary(task3_token_dictionary_);
        static_cast<SemanticDictionary *>(task3_dictionary_)->SetDependencyDictionary(task3_dependency_dictionary_);
        task3_dependency_dictionary_->CreateLabelDictionary(formalism, GetSemanticReader(formalism));
        static_cast<SemanticDictionary *>(task3_dictionary_)->CreatePredicateRoleDictionaries(formalism,
                                                                                              GetSemanticReader(
                                                                                                      formalism));
    }
}

void NeuralSemanticPipe::PreprocessData() {
    delete task1_token_dictionary_;
    delete task2_token_dictionary_;
    delete task3_token_dictionary_;
    CreateTokenDictionary();
    delete task1_dependency_dictionary_;
    delete task2_dependency_dictionary_;
    delete task3_dependency_dictionary_;
    CreateDependencyDictionary();

    PreprocessData("task1");
    PreprocessData("task2");
    PreprocessData("task3");
}

void NeuralSemanticPipe::ComputeScores(Instance *instance, Parts *parts,
                                       Features *features, bool pruner, vector<double> *scores) {
    Parameters *parameters;
    SemanticDictionary *semantic_dictionary =
            static_cast<SemanticDictionary *>(dictionary_);
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);
    if (pruner) {
        parameters = pruner_parameters_;
    } else {
        parameters = parameters_;
    }
    scores->resize(parts->size());
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    for (int r = 0; r < parts->size(); ++r) {
        bool has_unlabeled_features =
                (semantic_features->GetNumPartFeatures(r) > 0);
        bool has_labeled_features =
                (semantic_features->GetNumLabeledPartFeatures(r) > 0);

        if (pruner)
            CHECK((*parts)[r]->type() == SEMANTICPART_ARC ||
                  (*parts)[r]->type() == SEMANTICPART_PREDICATE);
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDARC) continue;
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDSIBLING) continue;

        // Compute scores for the unlabeled features.
        if (has_unlabeled_features) {
            const BinaryFeatures &part_features =
                    semantic_features->GetPartFeatures(r);
            (*scores)[r] = parameters->ComputeScore(part_features);
        } else {
            (*scores)[r] = 0.0;
        }

        // Compute scores for the labeled features.
        if ((*parts)[r]->type() == SEMANTICPART_ARC && !pruner &&
            GetSemanticOptions()->labeled()) {
            // Labeled arcs will be treated by looking at the unlabeled arcs and
            // conjoining with the label.
            CHECK(has_labeled_features);
            SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);

            const vector<int> &index_labeled_parts =
                    semantic_parts->FindLabeledArcs(arc->predicate(),
                                                    arc->argument(),
                                                    arc->sense());
            vector<int> allowed_labels(index_labeled_parts.size());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                CHECK_GE(index_labeled_parts[k], 0);
                CHECK_LT(index_labeled_parts[k], parts->size());
                SemanticPartLabeledArc *labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*parts)[index_labeled_parts[k]]);
                CHECK(labeled_arc != NULL);
                allowed_labels[k] = labeled_arc->role();
                LOG(INFO) << labeled_arc->predicate() << endl;
                LOG(INFO) << labeled_arc->argument() << endl;
                LOG(INFO) << labeled_arc->role() << endl;
            }
            vector<double> label_scores;
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            parameters->ComputeLabelScores(part_features, allowed_labels, &label_scores);
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                (*scores)[index_labeled_parts[k]] = label_scores[k];
            }
        } else if ((*parts)[r]->type() == SEMANTICPART_SIBLING &&
                   has_labeled_features) {
            // Labeled siblings will be treated by looking at the unlabeled ones and
            // conjoining with the label.
            CHECK(!pruner);
            CHECK(GetSemanticOptions()->labeled());
            SemanticPartSibling *sibling =
                    static_cast<SemanticPartSibling *>((*parts)[r]);
            const vector<int> &index_labeled_parts =
                    semantic_parts->GetLabeledParts(r);
            vector<int> bigram_labels(index_labeled_parts.size());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                CHECK_GE(index_labeled_parts[k], 0);
                CHECK_LT(index_labeled_parts[k], parts->size());
                SemanticPartLabeledSibling *labeled_sibling =
                        static_cast<SemanticPartLabeledSibling *>(
                                (*parts)[index_labeled_parts[k]]);
                CHECK(labeled_sibling != NULL);
                bigram_labels[k] = semantic_dictionary->GetRoleBigramLabel(
                        labeled_sibling->first_role(),
                        labeled_sibling->second_role());
            }
            vector<double> label_scores;
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            parameters->ComputeLabelScores(part_features, bigram_labels,
                                           &label_scores);
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                (*scores)[index_labeled_parts[k]] = label_scores[k];
            }
        }
    }
}

void NeuralSemanticPipe::RemoveUnsupportedFeatures(Instance *instance, Parts *parts, bool pruner,
                                                   const vector<bool> &selected_parts, Features *features) {
    Parameters *parameters;
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);
    if (pruner) {
        parameters = pruner_parameters_;
    } else {
        parameters = parameters_;
    }

    for (int r = 0; r < parts->size(); ++r) {
        // TODO: Make sure we can do this continue for the labeled parts...
        if (!selected_parts[r]) continue;

        bool has_unlabeled_features =
                (semantic_features->GetNumPartFeatures(r) > 0);
        bool has_labeled_features =
                (semantic_features->GetNumLabeledPartFeatures(r) > 0);

        if (pruner)
            CHECK((*parts)[r]->type() == SEMANTICPART_ARC ||
                  (*parts)[r]->type() == SEMANTICPART_PREDICATE);

        // TODO(atm): I think this is handling the case there can be labeled
        // features, but was never tested.
        CHECK(!has_labeled_features);

        // Skip labeled arcs, as they use the features from unlabeled arcs.
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDARC) continue;
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDSIBLING) continue;

        if (has_unlabeled_features) {
            BinaryFeatures *part_features =
                    semantic_features->GetMutablePartFeatures(r);
            int num_supported = 0;
            for (int j = 0; j < part_features->size(); ++j) {
                if (parameters->Exists((*part_features)[j])) {
                    (*part_features)[num_supported] = (*part_features)[j];
                    ++num_supported;
                }
            }
            part_features->resize(num_supported);
        }

        if (has_labeled_features) {
            BinaryFeatures *part_features =
                    semantic_features->GetMutableLabeledPartFeatures(r);
            int num_supported = 0;
            for (int j = 0; j < part_features->size(); ++j) {
                if (parameters->ExistsLabeled((*part_features)[j])) {
                    (*part_features)[num_supported] = (*part_features)[j];
                    ++num_supported;
                }
            }
            part_features->resize(num_supported);
        }
    }
}

void NeuralSemanticPipe::MakeGradientStep(Parts *parts, Features *features, double eta,
                                          int iteration, const vector<double> &gold_output,
                                          const vector<double> &predicted_output) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticDictionary *semantic_dictionary =
            static_cast<SemanticDictionary *>(dictionary_);
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);
    Parameters *parameters = GetTrainingParameters();

    for (int r = 0; r < parts->size(); ++r) {
        bool has_unlabeled_features =
                (semantic_features->GetNumPartFeatures(r) > 0);
        bool has_labeled_features =
                (semantic_features->GetNumLabeledPartFeatures(r) > 0);

        if ((*parts)[r]->type() == SEMANTICPART_LABELEDARC) continue;
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDSIBLING) continue;

        // Make updates for the unlabeled features.
        if (has_unlabeled_features) {
            if (predicted_output[r] != gold_output[r]) {
                const BinaryFeatures &part_features =
                        semantic_features->GetPartFeatures(r);
                parameters->MakeGradientStep(part_features, eta, iteration,
                                             predicted_output[r] - gold_output[r]);
            }
        }

        // Make updates for the labeled features.
        if ((*parts)[r]->type() == SEMANTICPART_ARC && has_labeled_features) {
            // Labeled arcs will be treated by looking at the unlabeled arcs and
            // conjoining with the label.
            CHECK(has_labeled_features);
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);
            const vector<int> &index_labeled_parts =
                    semantic_parts->FindLabeledArcs(arc->predicate(), arc->argument(), arc->sense());

            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledArc *labeled_arc =
                        static_cast<SemanticPartLabeledArc *>((*parts)[index_part]);
                CHECK(labeled_arc != NULL);
                double value = predicted_output[index_part] - gold_output[index_part];
                if (value != 0.0) {
                    parameters->MakeLabelGradientStep(part_features, eta, iteration,
                                                      labeled_arc->role(), value);
                }
            }
        } else if ((*parts)[r]->type() == SEMANTICPART_SIBLING && has_labeled_features) {
            // Labeled siblings will be treated by looking at the unlabeled ones and
            // conjoining with the label.
            CHECK(GetSemanticOptions()->labeled());
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            SemanticPartSibling *sibling =
                    static_cast<SemanticPartSibling *>((*parts)[r]);
            const vector<int> &index_labeled_parts =
                    semantic_parts->GetLabeledParts(r);
            vector<int> bigram_labels(index_labeled_parts.size());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledSibling *labeled_sibling =
                        static_cast<SemanticPartLabeledSibling *>((*parts)[index_part]);
                CHECK(labeled_sibling != NULL);
                int bigram_label = semantic_dictionary->GetRoleBigramLabel(
                        labeled_sibling->first_role(),
                        labeled_sibling->second_role());
                double value = predicted_output[index_part] - gold_output[index_part];
                if (value != 0.0) {
                    parameters->MakeLabelGradientStep(part_features, eta, iteration,
                                                      bigram_label, value);
                }
            }
        }
    }
}

void NeuralSemanticPipe::TouchParameters(Parts *parts, Features *features,
                                         const vector<bool> &selected_parts) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticDictionary *semantic_dictionary =
            static_cast<SemanticDictionary *>(dictionary_);
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);
    Parameters *parameters = GetTrainingParameters();

    for (int r = 0; r < parts->size(); ++r) {
        // TODO: Make sure we can do this continue for the labeled parts...
        if (!selected_parts[r]) continue;

        bool has_unlabeled_features =
                (semantic_features->GetNumPartFeatures(r) > 0);
        bool has_labeled_features =
                (semantic_features->GetNumLabeledPartFeatures(r) > 0);

        if ((*parts)[r]->type() == SEMANTICPART_LABELEDARC) continue;
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDSIBLING) continue;

        // Make updates for the unlabeled features.
        if (has_unlabeled_features) {
            const BinaryFeatures &part_features =
                    semantic_features->GetPartFeatures(r);
            parameters->MakeGradientStep(part_features, 0.0, 0, 0.0);
        }

        // Make updates for the labeled features.
        if ((*parts)[r]->type() == SEMANTICPART_ARC && has_labeled_features) {
            // Labeled arcs will be treated by looking at the unlabeled arcs and
            // conjoining with the label.
            CHECK(has_labeled_features);
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);
            const vector<int> &index_labeled_parts =
                    semantic_parts->FindLabeledArcs(arc->predicate(),
                                                    arc->argument(),
                                                    arc->sense());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledArc *labeled_arc =
                        static_cast<SemanticPartLabeledArc *>((*parts)[index_part]);
                CHECK(labeled_arc != NULL);
                parameters->MakeLabelGradientStep(part_features, 0.0, 0,
                                                  labeled_arc->role(), 0.0);
            }
        } else if ((*parts)[r]->type() == SEMANTICPART_SIBLING &&
                   has_labeled_features) {
            // Labeled siblings will be treated by looking at the unlabeled ones and
            // conjoining with the label.
            CHECK(GetSemanticOptions()->labeled());
            const BinaryFeatures &part_features = semantic_features->GetLabeledPartFeatures(r);
            SemanticPartSibling *sibling = static_cast<SemanticPartSibling *>((*parts)[r]);
            const vector<int> &index_labeled_parts = semantic_parts->GetLabeledParts(r);
            vector<int> bigram_labels(index_labeled_parts.size());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledSibling *labeled_sibling =
                        static_cast<SemanticPartLabeledSibling *>((*parts)[index_part]);
                CHECK(labeled_sibling != NULL);
                int bigram_label = semantic_dictionary->GetRoleBigramLabel(
                        labeled_sibling->first_role(), labeled_sibling->second_role());
                parameters->MakeLabelGradientStep(part_features, 0.0, 0,
                                                  bigram_label, 0.0);
            }
        }
    }
}

void NeuralSemanticPipe::MakeFeatureDifference(Parts *parts, Features *features,
                                               const vector<double> &gold_output,
                                               const vector<double> &predicted_output,
                                               FeatureVector *difference) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticDictionary *semantic_dictionary = static_cast<SemanticDictionary *>(dictionary_);
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);

    for (int r = 0; r < parts->size(); ++r) {
        bool has_unlabeled_features =
                (semantic_features->GetNumPartFeatures(r) > 0);
        bool has_labeled_features =
                (semantic_features->GetNumLabeledPartFeatures(r) > 0);

        if ((*parts)[r]->type() == SEMANTICPART_LABELEDARC) continue;
        if ((*parts)[r]->type() == SEMANTICPART_LABELEDSIBLING) continue;

        // Compute feature difference for the unlabeled features.
        if (has_unlabeled_features) {
            if (predicted_output[r] != gold_output[r]) {
                const BinaryFeatures &part_features =
                        semantic_features->GetPartFeatures(r);
                for (int j = 0; j < part_features.size(); ++j) {
                    difference->mutable_weights()->Add(part_features[j],
                                                       predicted_output[r] -
                                                       gold_output[r]);
                }
            }
        }

        // Make updates for the labeled features.
        if ((*parts)[r]->type() == SEMANTICPART_ARC && has_labeled_features) {
            // Labeled arcs will be treated by looking at the unlabeled arcs and
            // conjoining with the label.
            CHECK(has_labeled_features);
            const BinaryFeatures &part_features =
                    semantic_features->GetLabeledPartFeatures(r);
            SemanticPartArc *arc = static_cast<SemanticPartArc *>((*parts)[r]);
            const vector<int> &index_labeled_parts =
                    semantic_parts->FindLabeledArcs(arc->predicate(),
                                                    arc->argument(),
                                                    arc->sense());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledArc *labeled_arc =
                        static_cast<SemanticPartLabeledArc *>((*parts)[index_part]);
                CHECK(labeled_arc != NULL);
                double value = predicted_output[index_part] - gold_output[index_part];
                if (value != 0.0) {
                    for (int j = 0; j < part_features.size(); ++j) {
                        difference->mutable_labeled_weights()->Add(part_features[j],
                                                                   labeled_arc->role(),
                                                                   value);
                    }
                }
            }
        } else if ((*parts)[r]->type() == SEMANTICPART_SIBLING &&
                   has_labeled_features) {
            // Labeled siblings will be treated by looking at the unlabeled ones and
            // conjoining with the label.
            CHECK(GetSemanticOptions()->labeled());
            const BinaryFeatures &part_features = semantic_features->GetLabeledPartFeatures(r);
            SemanticPartSibling *sibling = static_cast<SemanticPartSibling *>((*parts)[r]);
            const vector<int> &index_labeled_parts = semantic_parts->GetLabeledParts(r);
            vector<int> bigram_labels(index_labeled_parts.size());
            for (int k = 0; k < index_labeled_parts.size(); ++k) {
                int index_part = index_labeled_parts[k];
                CHECK_GE(index_part, 0);
                CHECK_LT(index_part, parts->size());
                SemanticPartLabeledSibling *labeled_sibling =
                        static_cast<SemanticPartLabeledSibling *>(
                                (*parts)[index_part]);
                CHECK(labeled_sibling != NULL);
                int bigram_label = semantic_dictionary->GetRoleBigramLabel(
                        labeled_sibling->first_role(),
                        labeled_sibling->second_role());
                double value = predicted_output[index_part] - gold_output[index_part];
                if (value != 0.0) {
                    for (int j = 0; j < part_features.size(); ++j) {
                        difference->mutable_labeled_weights()->Add(part_features[j],
                                                                   bigram_label, value);
                    }
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakeParts(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    int sentence_length =
            static_cast<SemanticInstanceNumeric *>(instance)->size();
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    semantic_parts->Initialize();
    bool make_gold = (gold_outputs != NULL);
    if (make_gold) gold_outputs->clear();

    if (train_pruner_) {
        // For the pruner, make only unlabeled arc-factored and predicate parts and
        // compute indices.
        MakePartsBasic(instance, false, parts, gold_outputs);
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(sentence_length, false);
    } else {
        // Make arc-factored and predicate parts and compute indices.
        MakePartsBasic(instance, parts, gold_outputs);
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(sentence_length, GetSemanticOptions()->labeled());

        // Make global parts.
        MakePartsGlobal(instance, parts, gold_outputs);
        semantic_parts->BuildOffsets();
    }
}

void NeuralSemanticPipe::MakePartsBasic(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    int sentence_length =
            static_cast<SemanticInstanceNumeric *>(instance)->size();
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);

    MakePartsBasic(instance, false, parts, gold_outputs);
    semantic_parts->BuildOffsets();
    semantic_parts->BuildIndices(sentence_length, false);

    // Prune using a basic first-order model.
    if (GetSemanticOptions()->prune_basic()) {
        if (options_->train()) {
            Prune(instance, parts, gold_outputs, true);
        } else {
            Prune(instance, parts, gold_outputs, false);
        }
        semantic_parts->BuildOffsets();
        semantic_parts->BuildIndices(sentence_length, false);
    }

    if (GetSemanticOptions()->labeled()) {
        MakePartsBasic(instance, true, parts, gold_outputs);
    }
}

void NeuralSemanticPipe::MakePartsBasic(Instance *instance, bool add_labeled_parts, Parts *parts,
                                        vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    int sentence_length = sentence->size();
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
        for (int p = 0; p < sentence_length; ++p) {
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
    for (int p = 0; p < sentence_length; ++p) {
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
        for (int a = 1; a < sentence_length; ++a) {
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

void NeuralSemanticPipe::MakePartsArbitrarySiblings(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    //bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();

    // Siblings: (p,s,a1) and (p,s,a2).
    for (int p = 0; p < sentence_length; ++p) {
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
            for (int a1 = 1; a1 < sentence_length; ++a1) {
                int r1 = semantic_parts->FindArc(p, a1, s);
                if (r1 < 0) continue;
                for (int a2 = a1 + 1; a2 < sentence_length; ++a2) {
                    int r2 = semantic_parts->FindArc(p, a2, s);
                    if (r2 < 0) continue;
                    Part *part = semantic_parts->CreatePartSibling(p, s, a1, a2);
                    semantic_parts->AddPart(part);
                    if (make_gold) {
                        // Logical AND of the two individual arcs.
                        gold_outputs->push_back((*gold_outputs)[r1] * (*gold_outputs)[r2]);
                    }
                }
            }
        }
    }
}

void
NeuralSemanticPipe::MakePartsLabeledArbitrarySiblings(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();

    int offset, size;
    semantic_parts->GetOffsetSibling(&offset, &size);
    for (int r = offset; r < offset + size; ++r) {
        SemanticPartSibling *sibling =
                static_cast<SemanticPartSibling *>((*semantic_parts)[r]);
        int p = sibling->predicate();
        int s = sibling->sense();
        int a1 = sibling->first_argument();
        int a2 = sibling->second_argument();
        const vector<int> &labeled_first_arc_indices =
                semantic_parts->FindLabeledArcs(p, a1, s);
        const vector<int> &labeled_second_arc_indices =
                semantic_parts->FindLabeledArcs(p, a2, s);
        for (int k = 0; k < labeled_first_arc_indices.size(); ++k) {
            int r1 = labeled_first_arc_indices[k];
            SemanticPartLabeledArc *first_labeled_arc =
                    static_cast<SemanticPartLabeledArc *>((*semantic_parts)[r1]);
            int first_role = first_labeled_arc->role();
            for (int l = 0; l < labeled_second_arc_indices.size(); ++l) {
                int r2 = labeled_second_arc_indices[l];
                SemanticPartLabeledArc *second_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>((*semantic_parts)[r2]);
                int second_role = second_labeled_arc->role();
                // To keep the number of parts manageable, only create parts for:
                // - same role (a1 == a2);
                // - frequent role pairs.
                if (first_role != second_role &&
                    !semantic_dictionary->IsFrequentRolePair(first_role, second_role)) {
                    continue;
                }
                Part *part = semantic_parts->CreatePartLabeledSibling(p, s, a1, a2, first_role, second_role);
                semantic_parts->AddLabeledPart(part, r);
                if (make_gold) {
                    // Logical AND of the two individual labeled arcs.
                    gold_outputs->push_back((*gold_outputs)[r1] * (*gold_outputs)[r2]);
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakePartsConsecutiveSiblings(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    //bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();

    // Find the predicate parts (necessary to identify the gold predicate senses).
    // TODO: Replace this by semantic_parts->GetPredicateSenses(p) or something?
    int offset_predicate_parts, num_predicate_parts;
    semantic_parts->GetOffsetPredicate(&offset_predicate_parts,
                                       &num_predicate_parts);
    vector<vector<int> > predicate_part_indices(sentence_length);
    for (int r = 0; r < num_predicate_parts; ++r) {
        SemanticPartPredicate *predicate_part =
                static_cast<SemanticPartPredicate *>((*parts)[offset_predicate_parts + r]);
        predicate_part_indices[predicate_part->predicate()].
                push_back(offset_predicate_parts + r);
    }

    // Consecutive siblings: (p,s,a1) and (p,s,a2).
    for (int p = 0; p < sentence_length; ++p) {
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
        //const vector<int> &senses = semantic_parts->GetSenses(p);
        //CHECK_EQ(senses.size(), predicates->size());
        for (int s = 0; s < predicates->size(); ++s) {
            bool sense_active;
            bool first_arc_active;
            bool second_arc_active = false;
            bool arc_between;

            // Check if this is the correct sense.
            if (make_gold) {
                //int r = senses[s];
                int r = -1;
                for (int k = 0; k < predicate_part_indices[p].size(); ++k) {
                    r = predicate_part_indices[p][k];
                    SemanticPartPredicate *predicate_part =
                            static_cast<SemanticPartPredicate *>((*parts)[r]);
                    if (predicate_part->sense() == s) break;
                }
                CHECK_GE(r, 0);
                if (r >= 0 && NEARLY_EQ_TOL((*gold_outputs)[r], 1.0, 1e-9)) {
                    sense_active = true;
                } else {
                    sense_active = false;
                }
            }

            // Right side.
            // Allow self loops (a1 = p). We use a1 = p-1 to denote the special case
            // in which a2 is the first argument.
            for (int a1 = p - 1; a1 < sentence_length; ++a1) {
                int r1 = -1;
                if (a1 >= p) {
                    r1 = semantic_parts->FindArc(p, a1, s);
                    if (r1 < 0) continue;
                }

                if (make_gold) {
                    // Check if the first arc is active.
                    if (a1 < p || NEARLY_EQ_TOL((*gold_outputs)[r1], 1.0, 1e-9)) {
                        first_arc_active = true;
                    } else {
                        first_arc_active = false;
                    }
                    arc_between = false;
                }

                for (int a2 = a1 + 1; a2 <= sentence_length; ++a2) {
                    int r2 = -1;
                    if (a2 < sentence_length) {
                        r2 = semantic_parts->FindArc(p, a2, s);
                        if (r2 < 0) continue;
                    }
                    if (make_gold) {
                        // Check if the second arc is active.
                        if (a2 == sentence_length ||
                            NEARLY_EQ_TOL((*gold_outputs)[r2], 1.0, 1e-9)) {
                            second_arc_active = true;
                        } else {
                            second_arc_active = false;
                        }
                    }

                    Part *part = (a1 >= p) ?
                                 semantic_parts->CreatePartConsecutiveSibling(p, s, a1, a2) :
                                 semantic_parts->CreatePartConsecutiveSibling(p, s, -1, a2);
                    semantic_parts->AddPart(part);

                    if (make_gold) {
                        double value = 0.0;
                        if (sense_active && first_arc_active && second_arc_active &&
                            !arc_between) {
                            value = 1.0;
                            arc_between = true;
                        }
                        gold_outputs->push_back(value);
                    }
                }
            }

            // Left side.
            // NOTE: Self loops (a1 = p) are disabled on the left side, to prevent
            // having repeated parts. We use a1 = p+1 to denote the special case
            // in which a2 is the first argument.
            for (int a1 = p + 1; a1 >= 0; --a1) {
                int r1 = -1;
                if (a1 <= p) {
                    r1 = semantic_parts->FindArc(p, a1, s);
                    if (r1 < 0) continue;
                }
                if (a1 == p) continue; // See NOTE above.

                if (make_gold) {
                    // Check if the first arc is active.
                    if (a1 > p || NEARLY_EQ_TOL((*gold_outputs)[r1], 1.0, 1e-9)) {
                        first_arc_active = true;
                    } else {
                        first_arc_active = false;
                    }
                    arc_between = false;
                }

                for (int a2 = a1 - 1; a2 >= -1; --a2) {
                    int r2 = -1;
                    if (a2 > -1) {
                        r2 = semantic_parts->FindArc(p, a2, s);
                        if (r2 < 0) continue;
                    }
                    if (a2 == p) continue; // See NOTE above.

                    if (make_gold) {
                        // Check if the second arc is active.
                        if (a2 == -1 ||
                            NEARLY_EQ_TOL((*gold_outputs)[r2], 1.0, 1e-9)) {
                            second_arc_active = true;
                        } else {
                            second_arc_active = false;
                        }
                    }

                    Part *part = (a1 <= p) ?
                                 semantic_parts->CreatePartConsecutiveSibling(p, s, a1, a2) :
                                 semantic_parts->CreatePartConsecutiveSibling(p, s, -1, a2);
                    semantic_parts->AddPart(part);

                    if (make_gold) {
                        double value = 0.0;
                        if (sense_active && first_arc_active && second_arc_active &&
                            !arc_between) {
                            value = 1.0;
                            arc_between = true;
                        }
                        gold_outputs->push_back(value);
                    }
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakePartsGrandparents(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    //bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();

    // Grandparents: (g,t,p) and (p,s,a).
    for (int g = 0; g < sentence_length; ++g) {
        if (g == 0 && !allow_root_predicate) continue;
        int lemma_id_g = TOKEN_UNKNOWN;
        if (use_predicate_senses) {
            lemma_id_g = sentence->GetLemmaId(g);
            CHECK_GE(lemma_id_g, 0);
        }
        const vector<SemanticPredicate *> *predicates_g =
                &semantic_dictionary->GetLemmaPredicates(lemma_id_g);
        if (predicates_g->size() == 0 && allow_unseen_predicates) {
            predicates_g = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
        }
        for (int t = 0; t < predicates_g->size(); ++t) {
            for (int p = 1; p < sentence_length; ++p) {
                int r1 = semantic_parts->FindArc(g, p, t);
                if (r1 < 0) continue;
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
                    for (int a = 1; a < sentence_length; ++a) {
                        int r2 = semantic_parts->FindArc(p, a, s);
                        if (r2 < 0) continue;
                        Part *part = semantic_parts->CreatePartGrandparent(g, t, p, s, a);
                        semantic_parts->AddPart(part);
                        if (make_gold) {
                            // Logical AND of the two individual arcs.
                            gold_outputs->push_back((*gold_outputs)[r1] * (*gold_outputs)[r2]);
                        }
                    }
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakePartsCoparents(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    //bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();

    // Co-parents: (p1,s1,a) and (p2,s2,a).
    // First predicate.
    for (int p1 = 0; p1 < sentence_length; ++p1) {
        if (p1 == 0 && !allow_root_predicate) continue;
        int lemma_id_p1 = TOKEN_UNKNOWN;
        if (use_predicate_senses) {
            lemma_id_p1 = sentence->GetLemmaId(p1);
            CHECK_GE(lemma_id_p1, 0);
        }
        const vector<SemanticPredicate *> *predicates_p1 =
                &semantic_dictionary->GetLemmaPredicates(lemma_id_p1);
        if (predicates_p1->size() == 0 && allow_unseen_predicates) {
            predicates_p1 = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
        }
        for (int s1 = 0; s1 < predicates_p1->size(); ++s1) {
            // Second predicate.
            for (int p2 = p1 + 1; p2 < sentence_length; ++p2) {
                int lemma_id_p2 = TOKEN_UNKNOWN;
                if (use_predicate_senses) {
                    lemma_id_p2 = sentence->GetLemmaId(p2);
                    CHECK_GE(lemma_id_p2, 0);
                }
                const vector<SemanticPredicate *> *predicates_p2 =
                        &semantic_dictionary->GetLemmaPredicates(lemma_id_p2);
                if (predicates_p2->size() == 0 && allow_unseen_predicates) {
                    predicates_p2 = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
                }
                for (int s2 = 0; s2 < predicates_p2->size(); ++s2) {
                    // Common argument.
                    for (int a = 1; a < sentence_length; ++a) {
                        int r1 = semantic_parts->FindArc(p1, a, s1);
                        if (r1 < 0) continue;
                        int r2 = semantic_parts->FindArc(p2, a, s2);
                        if (r2 < 0) continue;
                        Part *part = semantic_parts->CreatePartCoparent(p1, s1, p2, s2, a);
                        semantic_parts->AddPart(part);
                        if (make_gold) {
                            // Logical AND of the two individual arcs.
                            gold_outputs->push_back((*gold_outputs)[r1] * (*gold_outputs)[r2]);
                        }
                    }
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakePartsConsecutiveCoparents(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int sentence_length = sentence->size();
    bool make_gold = (gold_outputs != NULL);
    SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
    SemanticOptions *semantic_options = GetSemanticOptions();
    //bool allow_self_loops = semantic_options->allow_self_loops();
    bool allow_root_predicate = semantic_options->allow_root_predicate();
    bool allow_unseen_predicates = semantic_options->allow_unseen_predicates();
    bool use_predicate_senses = semantic_options->use_predicate_senses();

    // Consecutive co-parents: (p1,s1,a) and (p2,s2,a).
    for (int a = 1; a < sentence_length; ++a) {
        bool first_arc_active;
        bool second_arc_active = false;
        bool arc_between;

        // Right side.
        // Allow self loops (p1 = a). We use p1 = a-1 to denote the special case
        // in which p2 is the first predicate.
        for (int p1 = a - 1; p1 < sentence_length; ++p1) {
            int num_senses1;
            if (p1 < a) {
                // If p1 = a-1, pretend there is a single sense (s1=0).
                num_senses1 = 1;
            } else {
                //const vector<int> &senses = semantic_parts->GetSenses(p);
                //CHECK_EQ(senses.size(), predicates->size());
                if (p1 == 0 && !allow_root_predicate) continue; // Never happens.
                int lemma_id = TOKEN_UNKNOWN;
                if (use_predicate_senses) {
                    lemma_id = sentence->GetLemmaId(p1);
                    CHECK_GE(lemma_id, 0);
                }
                const vector<SemanticPredicate *> *predicates =
                        &semantic_dictionary->GetLemmaPredicates(lemma_id);
                if (predicates->size() == 0 && allow_unseen_predicates) {
                    predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
                }
                num_senses1 = predicates->size();
            }

            for (int s1 = 0; s1 < num_senses1; ++s1) {
                int r1 = -1;
                if (p1 >= a) {
                    r1 = semantic_parts->FindArc(p1, a, s1);
                    if (r1 < 0) continue;
                }

                if (make_gold) {
                    // Check if the first arc is active.
                    if (p1 < a || NEARLY_EQ_TOL((*gold_outputs)[r1], 1.0, 1e-9)) {
                        first_arc_active = true;
                    } else {
                        first_arc_active = false;
                    }
                    arc_between = false;
                }

                for (int p2 = p1 + 1; p2 <= sentence_length; ++p2) {
                    int num_senses2;
                    if (p2 == sentence_length) {
                        // If p2 = sentence_length, pretend there is a single sense (s2=0).
                        num_senses2 = 1;
                    } else {
                        //const vector<int> &senses = semantic_parts->GetSenses(p);
                        //CHECK_EQ(senses.size(), predicates->size());
                        if (p2 == 0 && !allow_root_predicate) continue; // Never happens.
                        int lemma_id = TOKEN_UNKNOWN;
                        if (use_predicate_senses) {
                            lemma_id = sentence->GetLemmaId(p2);
                            CHECK_GE(lemma_id, 0);
                        }
                        const vector<SemanticPredicate *> *predicates =
                                &semantic_dictionary->GetLemmaPredicates(lemma_id);
                        if (predicates->size() == 0 && allow_unseen_predicates) {
                            predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
                        }
                        num_senses2 = predicates->size();
                    }

                    for (int s2 = 0; s2 < num_senses2; ++s2) {
                        int r2 = -1;
                        if (p2 < sentence_length) {
                            r2 = semantic_parts->FindArc(p2, a, s2);
                            if (r2 < 0) continue;
                        }
                        if (make_gold) {
                            // Check if the second arc is active.
                            if (p2 == sentence_length ||
                                NEARLY_EQ_TOL((*gold_outputs)[r2], 1.0, 1e-9)) {
                                second_arc_active = true;
                            } else {
                                second_arc_active = false;
                            }
                        }

                        Part *part = (p1 >= a) ?
                                     semantic_parts->CreatePartConsecutiveCoparent(p1, s1, p2, s2, a) :
                                     semantic_parts->CreatePartConsecutiveCoparent(-1, 0, p2, s2, a);
                        semantic_parts->AddPart(part);

                        if (make_gold) {
                            double value = 0.0;
                            if (first_arc_active && second_arc_active && !arc_between) {
                                value = 1.0;
                                arc_between = true;
                            }
                            gold_outputs->push_back(value);
                        }
                    }
                }
            }
        }

        // Left side.
        // NOTE: Self loops (p1 = a) are disabled on the left side, to prevent
        // having repeated parts. We use p1 = a+1 to denote the special case
        // in which p2 is the first predicate.
        for (int p1 = a + 1; p1 >= 0; --p1) {
            int num_senses1;
            if (p1 > a) {
                // If p1 = a+1, pretend there is a single sense (s1=0).
                num_senses1 = 1;
            } else if (p1 == a) { // See NOTE above.
                continue;
            } else {
                //const vector<int> &senses = semantic_parts->GetSenses(p);
                //CHECK_EQ(senses.size(), predicates->size());
                if (p1 == 0 && !allow_root_predicate) continue;
                int lemma_id = TOKEN_UNKNOWN;
                if (use_predicate_senses) {
                    lemma_id = sentence->GetLemmaId(p1);
                    CHECK_GE(lemma_id, 0);
                }
                const vector<SemanticPredicate *> *predicates =
                        &semantic_dictionary->GetLemmaPredicates(lemma_id);
                if (predicates->size() == 0 && allow_unseen_predicates) {
                    predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
                }
                num_senses1 = predicates->size();
            }

            for (int s1 = 0; s1 < num_senses1; ++s1) {
                int r1 = -1;
                if (p1 <= a) {
                    r1 = semantic_parts->FindArc(p1, a, s1);
                    if (r1 < 0) continue;
                }
                if (p1 == a) continue; // See NOTE above.

                if (make_gold) {
                    // Check if the first arc is active.
                    if (p1 > a || NEARLY_EQ_TOL((*gold_outputs)[r1], 1.0, 1e-9)) {
                        first_arc_active = true;
                    } else {
                        first_arc_active = false;
                    }
                    arc_between = false;
                }

                for (int p2 = p1 - 1; p2 >= -1; --p2) {
                    int num_senses2;
                    if (p2 == -1) {
                        // If p2 = -1, pretend there is a single sense (s2=0).
                        num_senses2 = 1;
                    } else if (p2 == a) { // See NOTE above.
                        continue;
                    } else {
                        //const vector<int> &senses = semantic_parts->GetSenses(p);
                        //CHECK_EQ(senses.size(), predicates->size());
                        if (p2 == 0 && !allow_root_predicate) continue;
                        int lemma_id = TOKEN_UNKNOWN;
                        if (use_predicate_senses) {
                            lemma_id = sentence->GetLemmaId(p2);
                            CHECK_GE(lemma_id, 0);
                        }
                        const vector<SemanticPredicate *> *predicates =
                                &semantic_dictionary->GetLemmaPredicates(lemma_id);
                        if (predicates->size() == 0 && allow_unseen_predicates) {
                            predicates = &semantic_dictionary->GetLemmaPredicates(TOKEN_UNKNOWN);
                        }
                        num_senses2 = predicates->size();
                    }

                    for (int s2 = 0; s2 < num_senses2; ++s2) {
                        int r2 = -1;
                        if (p2 > -1) {
                            r2 = semantic_parts->FindArc(p2, a, s2);
                            if (r2 < 0) continue;
                        }
                        if (p2 == a) continue; // See NOTE above.

                        if (make_gold) {
                            // Check if the second arc is active.
                            if (p2 == -1 ||
                                NEARLY_EQ_TOL((*gold_outputs)[r2], 1.0, 1e-9)) {
                                second_arc_active = true;
                            } else {
                                second_arc_active = false;
                            }
                        }

                        Part *part = (p1 <= a) ?
                                     semantic_parts->CreatePartConsecutiveCoparent(p1, s1, p2, s2, a) :
                                     semantic_parts->CreatePartConsecutiveCoparent(-1, 0, p2, s2, a);
                        semantic_parts->AddPart(part);

                        if (make_gold) {
                            double value = 0.0;
                            if (first_arc_active && second_arc_active && !arc_between) {
                                value = 1.0;
                                arc_between = true;
                            }
                            gold_outputs->push_back(value);
                        }
                    }
                }
            }
        }
    }
}

void NeuralSemanticPipe::MakePartsGlobal(Instance *instance, Parts *parts, vector<double> *gold_outputs) {
    SemanticOptions *semantic_options = GetSemanticOptions();
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);

    int num_parts_initial = semantic_parts->size();
    if (semantic_options->use_arbitrary_siblings()) {
        MakePartsArbitrarySiblings(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetSibling(num_parts_initial, semantic_parts->size() - num_parts_initial);
    //LOG(INFO) << "Num siblings: " << semantic_parts->size() - num_parts_initial;

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_arbitrary_siblings() &&
        FLAGS_use_labeled_sibling_features) {
        MakePartsLabeledArbitrarySiblings(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetLabeledSibling(
            num_parts_initial, semantic_parts->size() - num_parts_initial);
    //LOG(INFO) << "Num labeled siblings: " << semantic_parts->size() - num_parts_initial;

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_consecutive_siblings()) {
        MakePartsConsecutiveSiblings(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetConsecutiveSibling(num_parts_initial, semantic_parts->size() - num_parts_initial);

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_grandparents()) {
        MakePartsGrandparents(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetGrandparent(num_parts_initial, semantic_parts->size() - num_parts_initial);

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_coparents()) {
        MakePartsCoparents(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetCoparent(num_parts_initial, semantic_parts->size() - num_parts_initial);

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_consecutive_coparents()) {
        MakePartsConsecutiveCoparents(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetConsecutiveCoparent(num_parts_initial, semantic_parts->size() - num_parts_initial);

#if 0
    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_grandsiblings()) {
        MakePartsGrandSiblings(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetGrandSiblings(num_parts_initial,
                                                                                 semantic_parts->size() - num_parts_initial);

    num_parts_initial = semantic_parts->size();
    if (semantic_options->use_trisiblings()) {
        MakePartsTriSiblings(instance, parts, gold_outputs);
    }
    semantic_parts->SetOffsetTriSiblings(num_parts_initial,
                                                                             semantic_parts->size() - num_parts_initial);
#endif
}


void
NeuralSemanticPipe::MakePartsCrossFormHighOrder(Instance *task1_instance, Parts *task1_parts,
                                                vector<double> *task1_gold_outputs,
                                                Instance *task2_instance, Parts *task2_parts,
                                                vector<double> *task2_gold_outputs,
                                                Instance *task3_instance, Parts *task3_parts,
                                                vector<double> *task3_gold_outputs,
                                                Parts *global_parts, vector<double> *global_gold_outputs) {
    SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
    int task1_offset_predicate_parts, task1_num_predicate_parts;
    task1_semantic_parts->GetOffsetPredicate(&task1_offset_predicate_parts,
                                             &task1_num_predicate_parts);
    int task1_offset_arcs, task1_num_arcs;
    task1_semantic_parts->GetOffsetArc(&task1_offset_arcs, &task1_num_arcs);
    int task1_offset_labeled_arcs, task1_num_labeled_arcs;
    task1_semantic_parts->GetOffsetLabeledArc(&task1_offset_labeled_arcs,
                                              &task1_num_labeled_arcs);

    SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
    int task2_offset_predicate_parts, task2_num_predicate_parts;
    task2_semantic_parts->GetOffsetPredicate(&task2_offset_predicate_parts,
                                             &task2_num_predicate_parts);
    int task2_offset_arcs, task2_num_arcs;
    task2_semantic_parts->GetOffsetArc(&task2_offset_arcs, &task2_num_arcs);
    int task2_offset_labeled_arcs, task2_num_labeled_arcs;
    task2_semantic_parts->GetOffsetLabeledArc(&task2_offset_labeled_arcs,
                                              &task2_num_labeled_arcs);

    SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
    int task3_offset_predicate_parts, task3_num_predicate_parts;
    task3_semantic_parts->GetOffsetPredicate(&task3_offset_predicate_parts,
                                             &task3_num_predicate_parts);
    int task3_offset_arcs, task3_num_arcs;
    task3_semantic_parts->GetOffsetArc(&task3_offset_arcs, &task3_num_arcs);
    int task3_offset_labeled_arcs, task3_num_labeled_arcs;
    task3_semantic_parts->GetOffsetLabeledArc(&task3_offset_labeled_arcs,
                                              &task3_num_labeled_arcs);

    SemanticParts *semantic_parts = static_cast<SemanticParts *>(global_parts);
    semantic_parts->Initialize();
    global_gold_outputs->clear();
    SemanticOptions *semantic_options = GetSemanticOptions();
    int task1_num_roles = GetSemanticDictionary("task1")->GetNumRoles() + 1;
    int task2_num_roles = GetSemanticDictionary("task2")->GetNumRoles() + 1;
    int task3_num_roles = GetSemanticDictionary("task3")->GetNumRoles() + 1;

    int num_parts_initial = semantic_parts->size();

    // task1 vs task2
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);
        int p = arc->predicate();
        int a = arc->argument();
        int s = arc->sense();
        int task2_r = task2_semantic_parts->FindArc(p, a, s);
        if (task2_r < 0)
            continue;
        CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);
        Part *part = semantic_parts->CreatePartCrossForm2ndOrder(p, a, s, 1, 1, -1);
        semantic_parts->AddPart(part);
        if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_r + task1_offset_arcs], 1.0, 1e-6)
            && NEARLY_EQ_TOL((*task2_gold_outputs)[task2_r], 1.0, 1e-6)) {
            global_gold_outputs->push_back(1.0);
        } else {
            global_gold_outputs->push_back(0.0);
        }
    }

    // task1 vs task3
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);
        int p = arc->predicate();
        int a = arc->argument();
        int s = arc->sense();
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task3_r < 0)
            continue;
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);
        Part *part = semantic_parts->CreatePartCrossForm2ndOrder(p, a, s, 1, -1, 1);
        semantic_parts->AddPart(part);
        if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_r + task1_offset_arcs], 1.0, 1e-6)
            && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_r], 1.0, 1e-6)) {
            global_gold_outputs->push_back(1.0);
        } else {
            global_gold_outputs->push_back(0.0);
        }
    }

    // task2 vs task3
    for (int task2_r = 0; task2_r < task2_num_arcs; ++task2_r) {
        CHECK((*task2_parts)[task2_r + task2_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task2_parts)[task2_r + task2_offset_arcs]);
        int p = arc->predicate();
        int a = arc->argument();
        int s = arc->sense();
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task3_r < 0)
            continue;
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);
        Part *part = semantic_parts->CreatePartCrossForm2ndOrder(p, a, s, -1, 1, 1);
        semantic_parts->AddPart(part);
        if (NEARLY_EQ_TOL((*task2_gold_outputs)[task2_r + task2_offset_arcs], 1.0, 1e-6)
            && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_r], 1.0, 1e-6)) {
            global_gold_outputs->push_back(1.0);
        } else {
            global_gold_outputs->push_back(0.0);
        }
    }
    semantic_parts->SetOffsetCrossForm2ndOrder(num_parts_initial, semantic_parts->size() - num_parts_initial);

    num_parts_initial = semantic_parts->size();
    // task1 vs task2
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);

        int p = task1_arc->predicate();
        int a = task1_arc->argument();
        int s = task1_arc->sense();
        int task2_r = task2_semantic_parts->FindArc(p, a, s);
        if (task2_r < 0)
            continue;
        CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);

        const vector<int> &task1_index_labeled_parts =
                task1_semantic_parts->FindLabeledArcs(p, a, s);
        const vector<int> &task2_index_labeled_parts =
                task2_semantic_parts->FindLabeledArcs(p, a, s);

        for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
            CHECK_GE(task1_index_labeled_parts[task1_k], 0);
            CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
            CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
            SemanticPartLabeledArc *task1_labeled_arc =
                    static_cast<SemanticPartLabeledArc *>(
                            (*task1_parts)[task1_index_labeled_parts[task1_k]]);
            CHECK(task1_labeled_arc != NULL);
            int task1_role = task1_labeled_arc->role();

            for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
                CHECK_GE(task2_index_labeled_parts[task2_k], 0);
                CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
                CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
                SemanticPartLabeledArc *task2_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task2_parts)[task2_index_labeled_parts[task2_k]]);
                CHECK(task2_labeled_arc != NULL);
                int task2_role = task2_labeled_arc->role();
                int task3_role = -1;
                int idx = task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                if (!IsCrossFormLabelAllowed(idx)) continue;
                Part *part = semantic_parts->CreatePartCrossFormLabeled2ndOrder(p, a, s, task1_role, task2_role,
                                                                                task3_role);
                semantic_parts->AddPart(part);

                if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_index_labeled_parts[task1_k]], 1.0, 1e-6)
                    && NEARLY_EQ_TOL((*task2_gold_outputs)[task2_index_labeled_parts[task2_k]], 1.0, 1e-6)) {
                    global_gold_outputs->push_back(1.0);
                } else {
                    global_gold_outputs->push_back(0.0);
                }
            }
        }
    }

    // task1 vs task3
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);

        int p = task1_arc->predicate();
        int a = task1_arc->argument();
        int s = task1_arc->sense();
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task3_r < 0)
            continue;
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

        const vector<int> &task1_index_labeled_parts =
                task1_semantic_parts->FindLabeledArcs(p, a, s);
        const vector<int> &task3_index_labeled_parts =
                task3_semantic_parts->FindLabeledArcs(p, a, s);

        for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
            CHECK_GE(task1_index_labeled_parts[task1_k], 0);
            CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
            CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
            SemanticPartLabeledArc *task1_labeled_arc =
                    static_cast<SemanticPartLabeledArc *>(
                            (*task1_parts)[task1_index_labeled_parts[task1_k]]);
            CHECK(task1_labeled_arc != NULL);
            int task1_role = task1_labeled_arc->role();

            for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                SemanticPartLabeledArc *task3_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                CHECK(task3_labeled_arc != NULL);
                int task3_role = task3_labeled_arc->role();
                int task2_role = -1;
                int idx = task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                if (!IsCrossFormLabelAllowed(idx)) continue;
                Part *part = semantic_parts->CreatePartCrossFormLabeled2ndOrder(p, a, s, task1_role, task2_role,
                                                                                task3_role);
                semantic_parts->AddPart(part);

                if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_index_labeled_parts[task1_k]], 1.0, 1e-6)
                    && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_index_labeled_parts[task3_k]], 1.0, 1e-6)) {
                    global_gold_outputs->push_back(1.0);
                } else {
                    global_gold_outputs->push_back(0.0);
                }

            }
        }
    }

    // task2 vs task3
    for (int task2_r = 0; task2_r < task2_num_arcs; ++task2_r) {
        CHECK((*task2_parts)[task2_r + task2_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *task2_arc = static_cast<SemanticPartArc *>((*task2_parts)[task2_r + task2_offset_arcs]);

        int p = task2_arc->predicate();
        int a = task2_arc->argument();
        int s = task2_arc->sense();
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task3_r < 0)
            continue;
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

        const vector<int> &task2_index_labeled_parts =
                task2_semantic_parts->FindLabeledArcs(p, a, s);
        const vector<int> &task3_index_labeled_parts =
                task3_semantic_parts->FindLabeledArcs(p, a, s);

        for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
            CHECK_GE(task2_index_labeled_parts[task2_k], 0);
            CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
            CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
            SemanticPartLabeledArc *task2_labeled_arc =
                    static_cast<SemanticPartLabeledArc *>(
                            (*task2_parts)[task2_index_labeled_parts[task2_k]]);
            CHECK(task2_labeled_arc != NULL);
            int task2_role = task2_labeled_arc->role();

            for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                SemanticPartLabeledArc *task3_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                CHECK(task3_labeled_arc != NULL);
                int task3_role = task3_labeled_arc->role();
                int task1_role = -1;
                int idx = task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                if (!IsCrossFormLabelAllowed(idx)) continue;
                Part *part = semantic_parts->CreatePartCrossFormLabeled2ndOrder(p, a, s, task1_role, task2_role,
                                                                                task3_role);
                semantic_parts->AddPart(part);

                if (NEARLY_EQ_TOL((*task2_gold_outputs)[task2_index_labeled_parts[task2_k]], 1.0, 1e-6)
                    && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_index_labeled_parts[task3_k]], 1.0, 1e-6)) {
                    global_gold_outputs->push_back(1.0);
                } else {
                    global_gold_outputs->push_back(0.0);
                }
            }
        }
    }
    semantic_parts->SetOffsetCrossFormLabeled2ndOrder(num_parts_initial, semantic_parts->size() - num_parts_initial);
    num_parts_initial = semantic_parts->size();
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);
        int p = arc->predicate();
        int a = arc->argument();
        int s = arc->sense();
        int task2_r = task2_semantic_parts->FindArc(p, a, s);
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task2_r < 0 || task3_r < 0)
            continue;
        CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);
        Part *part = semantic_parts->CreatePartCrossForm3rdOrder(p, a, s, 1, 1, 1);
        semantic_parts->AddPart(part);
        if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_r + task1_offset_arcs], 1.0, 1e-6)
            && NEARLY_EQ_TOL((*task2_gold_outputs)[task2_r], 1.0, 1e-6)
            && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_r], 1.0, 1e-6)) {
            global_gold_outputs->push_back(1.0);
        } else {
            global_gold_outputs->push_back(0.0);
        }
    }
    semantic_parts->SetOffsetCrossForm3rdOrder(num_parts_initial, semantic_parts->size() - num_parts_initial);

    // task1 vs task2 vs task3
    num_parts_initial = semantic_parts->size();
    for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
        CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
        SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);

        int p = task1_arc->predicate();
        int a = task1_arc->argument();
        int s = task1_arc->sense();
        int task2_r = task2_semantic_parts->FindArc(p, a, s);
        int task3_r = task3_semantic_parts->FindArc(p, a, s);
        if (task2_r < 0 || task3_r < 0)
            continue;
        CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);
        CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

        const vector<int> &task1_index_labeled_parts =
                task1_semantic_parts->FindLabeledArcs(p, a, s);
        const vector<int> &task2_index_labeled_parts =
                task2_semantic_parts->FindLabeledArcs(p, a, s);
        const vector<int> &task3_index_labeled_parts =
                task3_semantic_parts->FindLabeledArcs(p, a, s);

        for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
            CHECK_GE(task1_index_labeled_parts[task1_k], 0);
            CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
            CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
            SemanticPartLabeledArc *task1_labeled_arc =
                    static_cast<SemanticPartLabeledArc *>(
                            (*task1_parts)[task1_index_labeled_parts[task1_k]]);
            CHECK(task1_labeled_arc != NULL);
            int task1_role = task1_labeled_arc->role();

            for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
                CHECK_GE(task2_index_labeled_parts[task2_k], 0);
                CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
                CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
                SemanticPartLabeledArc *task2_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task2_parts)[task2_index_labeled_parts[task2_k]]);
                CHECK(task2_labeled_arc != NULL);
                int task2_role = task2_labeled_arc->role();

                for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                    CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                    CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                    SemanticPartLabeledArc *task3_labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                    CHECK(task3_labeled_arc != NULL);
                    int task3_role = task3_labeled_arc->role();
                    int idx =
                            task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                    if (!IsCrossFormLabelAllowed(idx)) continue;
                    Part *part = semantic_parts->CreatePartCrossFormLabeled3rdOrder(p, a, s, task1_role, task2_role,
                                                                                    task3_role);
                    semantic_parts->AddPart(part);

                    if (NEARLY_EQ_TOL((*task1_gold_outputs)[task1_index_labeled_parts[task1_k]], 1.0, 1e-6)
                        && NEARLY_EQ_TOL((*task2_gold_outputs)[task2_index_labeled_parts[task2_k]], 1.0, 1e-6)
                        && NEARLY_EQ_TOL((*task3_gold_outputs)[task3_index_labeled_parts[task3_k]], 1.0, 1e-6)) {
                        global_gold_outputs->push_back(1.0);
                    } else {
                        global_gold_outputs->push_back(0.0);
                    }
                }
            }
        }
    }
    semantic_parts->SetOffsetCrossFormLabeled3rdOrder(num_parts_initial, semantic_parts->size() - num_parts_initial);
}

void NeuralSemanticPipe::MakeSelectedFeatures(Instance *instance,
                                              Parts *parts,
                                              bool pruner,
                                              const vector<bool> &selected_parts,
                                              Features *features) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticFeatures *semantic_features =
            static_cast<SemanticFeatures *>(features);
    int sentence_length = sentence->size();

    semantic_features->Initialize(instance, parts);

    // Build features for predicates.
    int offset, size;
    semantic_parts->GetOffsetPredicate(&offset, &size);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartPredicate *predicate_part =
                static_cast<SemanticPartPredicate *>((*semantic_parts)[r]);
        // Get the predicate id for this part.
        // TODO(atm): store this somewhere, so that we don't need to recompute this
        // all the time.
        int lemma_id = TOKEN_UNKNOWN;
        if (GetSemanticOptions()->use_predicate_senses()) {
            lemma_id = sentence->GetLemmaId(predicate_part->predicate());
        }
        const vector<SemanticPredicate *> *predicates =
                &GetSemanticDictionary()->GetLemmaPredicates(lemma_id);
        if (predicates->size() == 0 &&
            GetSemanticOptions()->allow_unseen_predicates()) {
            predicates = &GetSemanticDictionary()->GetLemmaPredicates(TOKEN_UNKNOWN);
        }
        int predicate_id = (*predicates)[predicate_part->sense()]->id();
        // Add the predicate features.
        semantic_features->AddPredicateFeatures(sentence, r,
                                                predicate_part->predicate(),
                                                predicate_id);
    }

    // Even in the case of labeled parsing, build features for unlabeled arcs
    // only. They will later be conjoined with the labels.
    semantic_parts->GetOffsetArc(&offset, &size);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartArc *arc =
                static_cast<SemanticPartArc *>((*semantic_parts)[r]);
        // Get the predicate id for this part.
        // TODO(atm): store this somewhere, so that we don't need to recompute this
        // all the time. Maybe store this directly in arc->sense()?
        int lemma_id = TOKEN_UNKNOWN;
        if (GetSemanticOptions()->use_predicate_senses()) {
            lemma_id = sentence->GetLemmaId(arc->predicate());
        }
        const vector<SemanticPredicate *> *predicates =
                &GetSemanticDictionary()->GetLemmaPredicates(lemma_id);
        if (predicates->size() == 0 &&
            GetSemanticOptions()->allow_unseen_predicates()) {
            predicates = &GetSemanticDictionary()->GetLemmaPredicates(TOKEN_UNKNOWN);
        }
        int predicate_id = (*predicates)[arc->sense()]->id();
        if (!pruner && GetSemanticOptions()->labeled()) {
            semantic_features->AddLabeledArcFeatures(sentence, r, arc->predicate(),
                                                     arc->argument(), predicate_id);
            if (!FLAGS_use_only_labeled_arc_features) {
                semantic_features->AddArcFeatures(sentence, r, arc->predicate(),
                                                  arc->argument(), predicate_id);
            }
        } else {
            semantic_features->AddArcFeatures(sentence, r, arc->predicate(),
                                              arc->argument(), predicate_id);
        }
    }

    // Build features for arbitrary siblings.
    semantic_parts->GetOffsetSibling(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartSibling *part =
                static_cast<SemanticPartSibling *>((*semantic_parts)[r]);
        CHECK_EQ(part->type(), SEMANTICPART_SIBLING);
        if (FLAGS_use_labeled_sibling_features) {
            semantic_features->
                    AddArbitraryLabeledSiblingFeatures(sentence, r,
                                                       part->predicate(),
                                                       part->sense(),
                                                       part->first_argument(),
                                                       part->second_argument());
            if (!FLAGS_use_only_labeled_sibling_features) {
                semantic_features->AddArbitrarySiblingFeatures(sentence, r,
                                                               part->predicate(),
                                                               part->sense(),
                                                               part->first_argument(),
                                                               part->second_argument());
            }
        } else {
            semantic_features->AddArbitrarySiblingFeatures(sentence, r,
                                                           part->predicate(),
                                                           part->sense(),
                                                           part->first_argument(),
                                                           part->second_argument());
        }
    }

    // Build features for consecutive siblings.
    semantic_parts->GetOffsetConsecutiveSibling(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartConsecutiveSibling *part =
                static_cast<SemanticPartConsecutiveSibling *>((*semantic_parts)[r]);
        CHECK_EQ(part->type(), SEMANTICPART_CONSECUTIVESIBLING);
        semantic_features->AddConsecutiveSiblingFeatures(
                sentence, r,
                part->predicate(),
                part->sense(),
                part->first_argument(),
                part->second_argument());
    }

    // Build features for grandparents.
    semantic_parts->GetOffsetGrandparent(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartGrandparent *part =
                static_cast<SemanticPartGrandparent *>((*semantic_parts)[r]);
        CHECK_EQ(part->type(), SEMANTICPART_GRANDPARENT);
        semantic_features->AddGrandparentFeatures(sentence, r,
                                                  part->grandparent_predicate(),
                                                  part->grandparent_sense(),
                                                  part->predicate(),
                                                  part->sense(),
                                                  part->argument());
    }

    // Build features for co-parents.
    semantic_parts->GetOffsetCoparent(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartCoparent *part =
                static_cast<SemanticPartCoparent *>((*semantic_parts)[r]);
        CHECK_EQ(part->type(), SEMANTICPART_COPARENT);
        semantic_features->AddCoparentFeatures(sentence, r,
                                               part->first_predicate(),
                                               part->first_sense(),
                                               part->second_predicate(),
                                               part->second_sense(),
                                               part->argument());
    }

    // Build features for consecutive co-parents.
    semantic_parts->GetOffsetConsecutiveCoparent(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
        if (!selected_parts[r]) continue;
        SemanticPartConsecutiveCoparent *part =
                static_cast<SemanticPartConsecutiveCoparent *>((*semantic_parts)[r]);
        CHECK_EQ(part->type(), SEMANTICPART_CONSECUTIVECOPARENT);
        semantic_features->AddConsecutiveCoparentFeatures(
                sentence, r,
                part->first_predicate(),
                part->first_sense(),
                part->second_predicate(),
                part->second_sense(),
                part->argument());
    }

#if 0
    // Build features for grand-siblings.
    dependency_parts->GetOffsetGrandSibl(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
      if (!selected_parts[r]) continue;
      SemanticPartGrandSibl *part =
        static_cast<SemanticPartGrandSibl*>((*dependency_parts)[r]);
      CHECK_EQ(part->type(), DEPENDENCYPART_GRANDSIBL);
      CHECK_LE(part->modifier(), sentence_length);
      CHECK_LE(part->sibling(), sentence_length);
      dependency_features->AddGrandSiblingFeatures(sentence, r,
                                                   part->grandparent(),
                                                   part->head(),
                                                   part->modifier(),
                                                   part->sibling());
    }

    // Build features for tri-siblings.
    dependency_parts->GetOffsetTriSibl(&offset, &size);
    if (pruner) CHECK_EQ(size, 0);
    for (int r = offset; r < offset + size; ++r) {
      if (!selected_parts[r]) continue;
      SemanticPartTriSibl *part =
        static_cast<SemanticPartTriSibl*>((*dependency_parts)[r]);
      CHECK_EQ(part->type(), DEPENDENCYPART_TRISIBL);
      dependency_features->AddTriSiblingFeatures(sentence, r,
                                                 part->head(),
                                                 part->modifier(),
                                                 part->sibling(),
                                                 part->other_sibling());
    }

#endif
}
// Prune basic parts (arcs and labeled arcs) using a first-order model.
// The vectors of basic parts is given as input, and those elements that are
// to be pruned are deleted from the vector.
// If gold_outputs is not NULL that vector will also be pruned.

void NeuralSemanticPipe::Prune(Instance *instance, Parts *parts,
                               vector<double> *gold_outputs,
                               bool preserve_gold) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    Features *features = CreateFeatures();
    vector<double> scores;
    vector<double> predicted_outputs;

    // Make sure gold parts are only preserved at training time.
    CHECK(!preserve_gold || options_->train());

    MakeFeatures(instance, parts, true, features);
    ComputeScores(instance, parts, features, true, &scores);
    GetSemanticDecoder()->DecodePruner(instance, parts, scores,
                                       &predicted_outputs);

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

    delete features;
}


void NeuralSemanticPipe::LabelInstance(Parts *parts, const vector<double> &output, Instance *instance) {
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    SemanticInstance *semantic_instance =
            static_cast<SemanticInstance *>(instance);
    SemanticDictionary *semantic_dictionary =
            static_cast<SemanticDictionary *>(dictionary_);
    string role =
            semantic_dictionary->GetRoleName(0);
    //LOG(INFO) << role<<endl;
    //bool allow_root_predicate = GetSemanticOptions()->allow_root_predicate();
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

void NeuralSemanticPipe::PruneCrossForm(const vector<int> &idxs) {
    Instance *task1_instance;
    vector<double> task1_scores;
    vector<double> task1_gold_outputs;
    vector<double> task1_predicted_outputs;
    Parts *task1_parts = CreateParts();

    Instance *task2_instance;
    vector<double> task2_scores;
    vector<double> task2_gold_outputs;
    vector<double> task2_predicted_outputs;
    Parts *task2_parts = CreateParts();

    Instance *task3_instance;
    vector<double> task3_scores;
    vector<double> task3_gold_outputs;
    vector<double> task3_predicted_outputs;
    Parts *task3_parts = CreateParts();

    SemanticOptions *semantic_options = GetSemanticOptions();
    task1_dictionary_->StopGrowth();
    task2_dictionary_->StopGrowth();
    task3_dictionary_->StopGrowth();
    int task1_num_roles = GetSemanticDictionary("task1")->GetNumRoles() + 1;
    int task2_num_roles = GetSemanticDictionary("task2")->GetNumRoles() + 1;
    int task3_num_roles = GetSemanticDictionary("task3")->GetNumRoles() + 1;

    for (int i = 0; i < idxs.size(); i++) {
        task1_instance = task1_instances_[idxs[i]];
        SetSemanticDictionary("task1");
        SetPruner("task1");
        MakeParts(task1_instance, task1_parts, &task1_gold_outputs);
        task2_instance = task2_instances_[idxs[i]];
        SetSemanticDictionary("task2");
        SetPruner("task2");
        MakeParts(task2_instance, task2_parts, &task2_gold_outputs);
        task3_instance = task3_instances_[idxs[i]];
        SetSemanticDictionary("task3");
        SetPruner("task3");
        MakeParts(task3_instance, task3_parts, &task3_gold_outputs);

        SemanticParts *task1_semantic_parts = static_cast<SemanticParts *>(task1_parts);
        int task1_offset_predicate_parts, task1_num_predicate_parts;
        task1_semantic_parts->GetOffsetPredicate(&task1_offset_predicate_parts,
                                                 &task1_num_predicate_parts);
        int task1_offset_arcs, task1_num_arcs;
        task1_semantic_parts->GetOffsetArc(&task1_offset_arcs, &task1_num_arcs);
        int task1_offset_labeled_arcs, task1_num_labeled_arcs;
        task1_semantic_parts->GetOffsetLabeledArc(&task1_offset_labeled_arcs,
                                                  &task1_num_labeled_arcs);

        SemanticParts *task2_semantic_parts = static_cast<SemanticParts *>(task2_parts);
        int task2_offset_predicate_parts, task2_num_predicate_parts;
        task2_semantic_parts->GetOffsetPredicate(&task2_offset_predicate_parts,
                                                 &task2_num_predicate_parts);
        int task2_offset_arcs, task2_num_arcs;
        task2_semantic_parts->GetOffsetArc(&task2_offset_arcs, &task2_num_arcs);
        int task2_offset_labeled_arcs, task2_num_labeled_arcs;
        task2_semantic_parts->GetOffsetLabeledArc(&task2_offset_labeled_arcs,
                                                  &task2_num_labeled_arcs);

        SemanticParts *task3_semantic_parts = static_cast<SemanticParts *>(task3_parts);
        int task3_offset_predicate_parts, task3_num_predicate_parts;
        task3_semantic_parts->GetOffsetPredicate(&task3_offset_predicate_parts,
                                                 &task3_num_predicate_parts);
        int task3_offset_arcs, task3_num_arcs;
        task3_semantic_parts->GetOffsetArc(&task3_offset_arcs, &task3_num_arcs);
        int task3_offset_labeled_arcs, task3_num_labeled_arcs;
        task3_semantic_parts->GetOffsetLabeledArc(&task3_offset_labeled_arcs,
                                                  &task3_num_labeled_arcs);


        // task1 vs task2
        for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
            CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
            SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);

            int p = task1_arc->predicate();
            int a = task1_arc->argument();
            int s = task1_arc->sense();
            int task2_r = task2_semantic_parts->FindArc(p, a, s);
            if (task2_r < 0)
                continue;
            CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);

            const vector<int> &task1_index_labeled_parts =
                    task1_semantic_parts->FindLabeledArcs(p, a, s);
            const vector<int> &task2_index_labeled_parts =
                    task2_semantic_parts->FindLabeledArcs(p, a, s);

            for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
                CHECK_GE(task1_index_labeled_parts[task1_k], 0);
                CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
                CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
                if (!NEARLY_EQ_TOL(task1_gold_outputs[task1_index_labeled_parts[task1_k]], 1.0, 1e-6))
                    continue;
                SemanticPartLabeledArc *task1_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task1_parts)[task1_index_labeled_parts[task1_k]]);
                CHECK(task1_labeled_arc != NULL);
                int task1_role = task1_labeled_arc->role();

                for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
                    CHECK_GE(task2_index_labeled_parts[task2_k], 0);
                    CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
                    if (!NEARLY_EQ_TOL(task2_gold_outputs[task2_index_labeled_parts[task2_k]], 1.0, 1e-6))
                        continue;

                    SemanticPartLabeledArc *task2_labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[task2_index_labeled_parts[task2_k]]);
                    CHECK(task2_labeled_arc != NULL);
                    int task2_role = task2_labeled_arc->role();
                    int task3_role = -1;
                    int idx =
                            task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                    CrossFormLabelOn(idx);
                }
            }
        }

        // task1 vs task3
        for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
            CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
            SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);

            int p = task1_arc->predicate();
            int a = task1_arc->argument();
            int s = task1_arc->sense();
            int task3_r = task3_semantic_parts->FindArc(p, a, s);
            if (task3_r < 0)
                continue;
            CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

            const vector<int> &task1_index_labeled_parts =
                    task1_semantic_parts->FindLabeledArcs(p, a, s);
            const vector<int> &task3_index_labeled_parts =
                    task3_semantic_parts->FindLabeledArcs(p, a, s);

            for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
                CHECK_GE(task1_index_labeled_parts[task1_k], 0);
                CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
                CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
                if (!NEARLY_EQ_TOL(task1_gold_outputs[task1_index_labeled_parts[task1_k]], 1.0, 1e-6))
                    continue;
                SemanticPartLabeledArc *task1_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task1_parts)[task1_index_labeled_parts[task1_k]]);
                CHECK(task1_labeled_arc != NULL);
                int task1_role = task1_labeled_arc->role();

                for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                    CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                    CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                    if (!NEARLY_EQ_TOL(task3_gold_outputs[task3_index_labeled_parts[task3_k]], 1.0, 1e-6))
                        continue;
                    SemanticPartLabeledArc *task3_labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                    CHECK(task3_labeled_arc != NULL);
                    int task3_role = task3_labeled_arc->role();
                    int task2_role = -1;
                    int idx =
                            task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                    CrossFormLabelOn(idx);
                }
            }
        }

        // task2 vs task3
        for (int task2_r = 0; task2_r < task2_num_arcs; ++task2_r) {
            CHECK((*task2_parts)[task2_r + task2_offset_arcs]->type() == SEMANTICPART_ARC);
            SemanticPartArc *task2_arc = static_cast<SemanticPartArc *>((*task2_parts)[task2_r + task2_offset_arcs]);

            int p = task2_arc->predicate();
            int a = task2_arc->argument();
            int s = task2_arc->sense();
            int task3_r = task3_semantic_parts->FindArc(p, a, s);
            if (task3_r < 0)
                continue;
            CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

            const vector<int> &task2_index_labeled_parts =
                    task2_semantic_parts->FindLabeledArcs(p, a, s);
            const vector<int> &task3_index_labeled_parts =
                    task3_semantic_parts->FindLabeledArcs(p, a, s);

            for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
                CHECK_GE(task2_index_labeled_parts[task2_k], 0);
                CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
                CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
                if (!NEARLY_EQ_TOL(task2_gold_outputs[task2_index_labeled_parts[task2_k]], 1.0, 1e-6))
                    continue;
                SemanticPartLabeledArc *task2_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task2_parts)[task2_index_labeled_parts[task2_k]]);
                CHECK(task2_labeled_arc != NULL);
                int task2_role = task2_labeled_arc->role();

                for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                    CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                    CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                    CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                    if (!NEARLY_EQ_TOL(task3_gold_outputs[task3_index_labeled_parts[task3_k]], 1.0, 1e-6))
                        continue;
                    SemanticPartLabeledArc *task3_labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                    CHECK(task3_labeled_arc != NULL);
                    int task3_role = task3_labeled_arc->role();
                    int task1_role = -1;
                    int idx =
                            task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles + task3_role;
                    CrossFormLabelOn(idx);
                }
            }
        }
        // task1 vs task2 vs task3
        for (int task1_r = 0; task1_r < task1_num_arcs; ++task1_r) {
            CHECK((*task1_parts)[task1_r + task1_offset_arcs]->type() == SEMANTICPART_ARC);
            SemanticPartArc *task1_arc = static_cast<SemanticPartArc *>((*task1_parts)[task1_r + task1_offset_arcs]);
            int p = task1_arc->predicate();
            int a = task1_arc->argument();
            int s = task1_arc->sense();
            int task2_r = task2_semantic_parts->FindArc(p, a, s);
            int task3_r = task3_semantic_parts->FindArc(p, a, s);
            if (task2_r < 0 || task3_r < 0)
                continue;
            CHECK((*task2_parts)[task2_r]->type() == SEMANTICPART_ARC);
            CHECK((*task3_parts)[task3_r]->type() == SEMANTICPART_ARC);

            const vector<int> &task1_index_labeled_parts =
                    task1_semantic_parts->FindLabeledArcs(p, a, s);
            const vector<int> &task2_index_labeled_parts =
                    task2_semantic_parts->FindLabeledArcs(p, a, s);
            const vector<int> &task3_index_labeled_parts =
                    task3_semantic_parts->FindLabeledArcs(p, a, s);
            for (int task1_k = 0; task1_k < task1_index_labeled_parts.size(); ++task1_k) {
                CHECK_GE(task1_index_labeled_parts[task1_k], 0);
                CHECK_LT(task1_index_labeled_parts[task1_k], task1_parts->size());
                CHECK_EQ((*task1_parts)[task1_index_labeled_parts[task1_k]]->type(), SEMANTICPART_LABELEDARC);
                if (!NEARLY_EQ_TOL(task1_gold_outputs[task1_index_labeled_parts[task1_k]], 1.0, 1e-6))
                    continue;
                SemanticPartLabeledArc *task1_labeled_arc =
                        static_cast<SemanticPartLabeledArc *>(
                                (*task1_parts)[task1_index_labeled_parts[task1_k]]);
                CHECK(task1_labeled_arc != NULL);
                int task1_role = task1_labeled_arc->role();

                for (int task2_k = 0; task2_k < task2_index_labeled_parts.size(); ++task2_k) {
                    CHECK_GE(task2_index_labeled_parts[task2_k], 0);
                    CHECK_LT(task2_index_labeled_parts[task2_k], task2_parts->size());
                    CHECK_EQ((*task2_parts)[task2_index_labeled_parts[task2_k]]->type(), SEMANTICPART_LABELEDARC);
                    if (!NEARLY_EQ_TOL(task2_gold_outputs[task2_index_labeled_parts[task2_k]], 1.0, 1e-6))
                        continue;
                    SemanticPartLabeledArc *task2_labeled_arc =
                            static_cast<SemanticPartLabeledArc *>(
                                    (*task2_parts)[task2_index_labeled_parts[task2_k]]);
                    CHECK(task2_labeled_arc != NULL);
                    int task2_role = task2_labeled_arc->role();

                    for (int task3_k = 0; task3_k < task3_index_labeled_parts.size(); ++task3_k) {
                        CHECK_GE(task3_index_labeled_parts[task3_k], 0);
                        CHECK_LT(task3_index_labeled_parts[task3_k], task3_parts->size());
                        CHECK_EQ((*task3_parts)[task3_index_labeled_parts[task3_k]]->type(), SEMANTICPART_LABELEDARC);
                        if (!NEARLY_EQ_TOL(task3_gold_outputs[task3_index_labeled_parts[task3_k]], 1.0, 1e-6))
                            continue;
                        SemanticPartLabeledArc *task3_labeled_arc =
                                static_cast<SemanticPartLabeledArc *>(
                                        (*task3_parts)[task3_index_labeled_parts[task3_k]]);
                        CHECK(task3_labeled_arc != NULL);
                        int task3_role = task3_labeled_arc->role();
                        int idx = task1_role * task2_num_roles * task3_num_roles + task2_role * task3_num_roles +
                                  task3_role;
                        CrossFormLabelOn(idx);
                    }
                }
            }
        }
    }
    delete task1_parts;
    delete task2_parts;
    delete task3_parts;
}


// for pruner
void NeuralSemanticPipe::Train(const string &formalism) {
    if (formalism != "task1" && formalism != "task2" && formalism != "task3") {
        LOG(INFO) << "Unsupported formalism: " << formalism << ". Giving up..." << endl;
        CHECK(1 == 0);
    }

    delete task1_token_dictionary_;
    delete task2_token_dictionary_;
    delete task3_token_dictionary_;
    CreateTokenDictionary();
    delete task1_dependency_dictionary_;
    delete task2_dependency_dictionary_;
    delete task3_dependency_dictionary_;
    CreateDependencyDictionary();

    PreprocessData(formalism);

    SetSemanticDictionary(formalism);
    SetSemanticReader(formalism);
    CreateInstances(formalism);
    SetInstances(formalism);
    parameters_->Initialize(options_->use_averaging());

    if (options_->only_supported_features()) MakeSupportedParameters();

    for (int i = 0; i < options_->GetNumEpochs(); ++i) {
        TrainEpoch(i);
    }

    parameters_->Finalize(options_->GetNumEpochs() * instances_.size());
}


void NeuralSemanticPipe::NeuralTrain() {
    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_ctf = semantic_options->output_term() == "freda3" || semantic_options->output_term() == "shared3";
    PreprocessData();
    CreateInstances("task1");
    CreateInstances("task2");
    CreateInstances("task3");
    vector<int> idxs;
    for (int i = 0; i < task1_instances_.size(); ++i) {
        idxs.push_back(i);
    }
    NeuralInitialize();
    SaveModelFile();
    LoadModelFile();
    LoadPruner(semantic_options->GetPrunerModelFilePath("task1"), "task1");
    LoadPruner(semantic_options->GetPrunerModelFilePath("task2"), "task2");
    LoadPruner(semantic_options->GetPrunerModelFilePath("task3"), "task3");
    if (semantic_options->use_pretrained_embedding()) {
        LoadPretrainedEmbedding();
    }
    if (use_ctf)
        PruneCrossForm(idxs);
    double unlabeled_F1 = 0, labeled_F1 = 0, best_labeled_F1 = -1;
    for (int i = 0; i < options_->GetNumEpochs(); ++i) {
        semantic_options->train_on();
        random_shuffle(idxs.begin(), idxs.end());
        NeuralTrainEpoch(idxs, i);
        semantic_options->train_off();
        NeuralRun(unlabeled_F1, labeled_F1);
        if (labeled_F1 > best_labeled_F1 && labeled_F1 > 0.60) {
            SaveNueralModel();
            best_labeled_F1 = labeled_F1;
        }
    }
}

double NeuralSemanticPipe::NeuralTrainEpoch(const vector<int> &idxs, int epoch) {
    Instance *task1_instance;
    vector<double> task1_scores;
    vector<double> task1_gold_outputs;
    vector<double> task1_predicted_outputs;
    Parts *task1_parts = CreateParts();

    Instance *task2_instance;
    vector<double> task2_scores;
    vector<double> task2_gold_outputs;
    vector<double> task2_predicted_outputs;
    Parts *task2_parts = CreateParts();

    Instance *task3_instance;
    vector<double> task3_scores;
    vector<double> task3_gold_outputs;
    vector<double> task3_predicted_outputs;
    Parts *task3_parts = CreateParts();

    vector<double> global_scores;
    vector<double> global_gold_outputs;
    vector<double> global_predicted_outputs;
    Parts *global_parts = CreateParts();

    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_ctf = semantic_options->output_term() == "freda3" || semantic_options->output_term() == "shared3";
    bool use_word_dropout = semantic_options->use_word_dropout();
    float word_dropout_rate = semantic_options->word_dropout_rate();
    if (use_word_dropout) {
        CHECK(word_dropout_rate > 0.0 && word_dropout_rate < 1.0);
    }
    float forward_loss = 0.0;
    int num_instances = idxs.size();
    if (epoch == 0) {
        LOG(INFO) << "Number of instances: " << num_instances << endl;
    }
    task1_dictionary_->StopGrowth();
    task2_dictionary_->StopGrowth();
    task3_dictionary_->StopGrowth();
    LOG(INFO) << " Iteration #" << epoch + 1;
    for (int i = 0; i < idxs.size(); i++) {
        task1_instance = task1_instances_[idxs[i]];
        SetSemanticDictionary("task1");
        SetPruner("task1");
        MakeParts(task1_instance, task1_parts, &task1_gold_outputs);
        task2_instance = task2_instances_[idxs[i]];
        SetSemanticDictionary("task2");
        SetPruner("task2");
        MakeParts(task2_instance, task2_parts, &task2_gold_outputs);
        task3_instance = task3_instances_[idxs[i]];
        SetSemanticDictionary("task3");
        SetPruner("task3");
        MakeParts(task3_instance, task3_parts, &task3_gold_outputs);
        if (use_ctf)
            MakePartsCrossFormHighOrder(task1_instance, task1_parts, &task1_gold_outputs,
                                        task2_instance, task2_parts, &task2_gold_outputs,
                                        task3_instance, task3_parts, &task3_gold_outputs,
                                        global_parts, &global_gold_outputs);
        dynet::ComputationGraph cg;
        Expression ex_loss = parser->BuildGraph(task1_instance, decoder_, true,
                                                task1_parts, task1_scores, task1_gold_outputs, task1_predicted_outputs,
                                                task2_parts, task2_scores, task2_gold_outputs, task2_predicted_outputs,
                                                task3_parts, task3_scores, task3_gold_outputs, task3_predicted_outputs,
                                                global_parts, global_scores, global_gold_outputs,
                                                global_predicted_outputs,
                                                use_word_dropout, word_dropout_rate,
                                                form_count_, cg);
        double loss = dynet::as_scalar(cg.forward(ex_loss));
        int corr = 0;
        for (int r = 0; r < task1_parts->size(); ++r) {
            if (NEARLY_EQ_TOL(task1_gold_outputs[r], task1_predicted_outputs[r], 1e-6)) corr += 1;
            else break;
        }

        for (int r = 0; r < task2_parts->size(); ++r) {
            if (NEARLY_EQ_TOL(task2_gold_outputs[r], task2_predicted_outputs[r], 1e-6)) corr += 1;
            else break;
        }

        for (int r = 0; r < task3_parts->size(); ++r) {
            if (NEARLY_EQ_TOL(task3_gold_outputs[r], task3_predicted_outputs[r], 1e-6)) corr += 1;
            else break;
        }

        for (int r = 0; r < global_parts->size(); ++r) {
            if (NEARLY_EQ_TOL(global_gold_outputs[r], global_predicted_outputs[r], 1e-6)) corr += 1;
            else break;
        }
        if (corr < task1_parts->size() + task2_parts->size() + task3_parts->size() + global_parts->size()) {
            cg.backward(ex_loss);
            trainer->update(1.0);
        }
        loss = max(loss, 0.0);
        forward_loss += loss;
    }
    delete task1_parts;
    delete task2_parts;
    delete task3_parts;
    delete global_parts;
    if (semantic_options->trainer() == "adam" && (epoch + 1) % 10 == 0)
        trainer->eta0 *= 0.5;
    trainer->update_epoch();
    forward_loss /= idxs.size();
    LOG(INFO) << "training loss: " << forward_loss << endl;
    trainer->status();
    return forward_loss;
}

void NeuralSemanticPipe::NeuralTest() {
    PreprocessData();
    LoadModelFile();
    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_ctf = semantic_options->output_term() == "freda3" || semantic_options->output_term() == "shared3";
    if (use_ctf) {
        CreateInstances("task1");
        CreateInstances("task2");
        CreateInstances("task3");
        vector<int> idxs;
        for (int r = 0; r < task1_instances_.size(); ++r)
            idxs.push_back(r);
        PruneCrossForm(idxs);
    }
    LoadNueralModel();
    LoadPruner(semantic_options->GetPrunerModelFilePath("task1"), "task1");
    LoadPruner(semantic_options->GetPrunerModelFilePath("task2"), "task2");
    LoadPruner(semantic_options->GetPrunerModelFilePath("task3"), "task3");
    semantic_options->train_off();
    double unlabeled_F1 = 0, labeled_F1 = 0;
    NeuralRun(unlabeled_F1, labeled_F1);
}

void NeuralSemanticPipe::NeuralRun(double &unlabeled_F1, double &labeled_F1) {
    Instance *task1_instance;
    vector<double> task1_scores;
    vector<double> task1_gold_outputs;
    vector<double> task1_predicted_outputs;
    Parts *task1_parts = CreateParts();

    Instance *task2_instance;
    vector<double> task2_scores;
    vector<double> task2_gold_outputs;
    vector<double> task2_predicted_outputs;
    Parts *task2_parts = CreateParts();

    Instance *task3_instance;
    vector<double> task3_scores;
    vector<double> task3_gold_outputs;
    vector<double> task3_predicted_outputs;
    Parts *task3_parts = CreateParts();

    vector<double> global_scores;
    vector<double> global_gold_outputs;
    vector<double> global_predicted_outputs;
    Parts *global_parts = CreateParts();

    timeval start, end;
    gettimeofday(&start, NULL);

    if (options_->evaluate()) BeginEvaluation();
    SemanticOptions *semantic_options = GetSemanticOptions();
    bool use_ctf = semantic_options->output_term() == "freda3" || semantic_options->output_term() == "shared3";

    task1_reader_->Open(semantic_options->GetTestFilePath("task1"));
    task1_writer_->Open(semantic_options->GetOutputFilePath("task1"));
    task2_reader_->Open(semantic_options->GetTestFilePath("task2"));
    task2_writer_->Open(semantic_options->GetOutputFilePath("task2"));
    task3_reader_->Open(semantic_options->GetTestFilePath("task3"));
    task3_writer_->Open(semantic_options->GetOutputFilePath("task3"));

    int num_instances = 0;

    task1_instance = task1_reader_->GetNext();
    task2_instance = task2_reader_->GetNext();
    task3_instance = task3_reader_->GetNext();

    double forward_loss = 0.0;

    while (task1_instance) {
        Instance *task1_formatted_instance = GetFormattedInstance("task1", task1_instance);
        SetSemanticDictionary("task1");
        SetPruner("task1");
        MakeParts(task1_formatted_instance, task1_parts, &task1_gold_outputs);
        Instance *task2_formatted_instance = GetFormattedInstance("task2", task2_instance);
        SetSemanticDictionary("task2");
        SetPruner("task2");
        MakeParts(task2_formatted_instance, task2_parts, &task2_gold_outputs);
        Instance *task3_formatted_instance = GetFormattedInstance("task3", task3_instance);
        SetSemanticDictionary("task3");
        SetPruner("task3");
        MakeParts(task3_formatted_instance, task3_parts, &task3_gold_outputs);
        if (use_ctf)
            MakePartsCrossFormHighOrder(task1_formatted_instance, task1_parts, &task1_gold_outputs,
                                        task2_formatted_instance, task2_parts, &task2_gold_outputs,
                                        task3_formatted_instance, task3_parts, &task3_gold_outputs,
                                        global_parts, &global_gold_outputs);

        dynet::ComputationGraph cg;
        Expression ex_loss = parser->BuildGraph(task1_formatted_instance, decoder_, false,
                                                task1_parts, task1_scores, task1_gold_outputs, task1_predicted_outputs,
                                                task2_parts, task2_scores, task2_gold_outputs, task2_predicted_outputs,
                                                task3_parts, task3_scores, task3_gold_outputs, task3_predicted_outputs,
                                                global_parts, global_scores, global_gold_outputs,
                                                global_predicted_outputs,
                                                false, 0.0,
                                                form_count_, cg);
        double loss = dynet::as_scalar(cg.forward(ex_loss));
        loss = max(loss, 0.0);
        forward_loss += loss;
        Instance *task1_output_instance = task1_instance->Copy();
        Instance *task2_output_instance = task2_instance->Copy();
        Instance *task3_output_instance = task3_instance->Copy();
        SetSemanticDictionary("task1");
        LabelInstance(task1_parts, task1_predicted_outputs, task1_output_instance);
        SetSemanticDictionary("task2");
        LabelInstance(task2_parts, task2_predicted_outputs, task2_output_instance);
        SetSemanticDictionary("task3");
        LabelInstance(task3_parts, task3_predicted_outputs, task3_output_instance);

        if (options_->evaluate()) {
            EvaluateInstance("task1", task1_instance, task1_output_instance,
                             task1_parts, task1_gold_outputs, task1_predicted_outputs);
            EvaluateInstance("task2", task2_instance, task2_output_instance,
                             task2_parts, task2_gold_outputs, task2_predicted_outputs);
            EvaluateInstance("task3", task3_instance, task3_output_instance,
                             task3_parts, task3_gold_outputs, task3_predicted_outputs);
        }

        task1_writer_->Write(task1_output_instance);
        task2_writer_->Write(task2_output_instance);
        task3_writer_->Write(task3_output_instance);
        if (task1_formatted_instance != task1_instance) delete task1_formatted_instance;
        if (task2_formatted_instance != task2_instance) delete task2_formatted_instance;
        if (task3_formatted_instance != task3_instance) delete task3_formatted_instance;
        delete task1_output_instance;
        delete task2_output_instance;
        delete task3_output_instance;
        delete task1_instance;
        delete task2_instance;
        delete task3_instance;

        task1_instance = task1_reader_->GetNext();
        task2_instance = task2_reader_->GetNext();
        task3_instance = task3_reader_->GetNext();
        ++num_instances;

    }
    forward_loss /= num_instances;
    delete task1_parts;
    delete task2_parts;
    delete task3_parts;
    delete global_parts;

    task1_writer_->Close();
    task1_reader_->Close();
    task2_writer_->Close();
    task2_reader_->Close();
    task3_writer_->Close();
    task3_reader_->Close();
    LOG(INFO) << "dev loss: " << forward_loss << endl;
    gettimeofday(&end, NULL);

#if USE_WEIGHT_CACHING == 1
    LOG(INFO) << "Cache size: " << parameters_->GetCachingWeightsSize() << "\t"
      << "Cache hits: " << parameters_->GetCachingWeightsHits() << "\t"
      << "Cache misses: " << parameters_->GetCachingWeightsMisses() << endl;
#endif

    if (options_->evaluate()) EndEvaluation(unlabeled_F1, labeled_F1);
}

void NeuralSemanticPipe::LoadPretrainedEmbedding() {
    SemanticOptions *semantic_option = GetSemanticOptions();
    embedding_ = new unordered_map<int, vector<float>>();
    form_count_ = new unordered_map<int, int>();
    unsigned dim = semantic_option->pre_word_dim();
    ifstream in(semantic_option->GetPretrainedEmbeddingFilePath());

    if (!in.is_open()) {
        cerr << "Pretrained embeddings FILE NOT FOUND!" << endl;
    }
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
        int form_id = task1_token_dictionary_->GetFormId(word);
        if (form_id < 0)
            continue;
        found += 1;

        (*embedding_)[form_id] = v;
    }
    in.close();
    LOG(INFO) << found << "/" << task1_token_dictionary_->GetNumForms() << " words found in the pretrained embedding"
              << endl;
    parser->LoadEmbedding(embedding_);
    delete embedding_;

    for (int i = 0; i < task1_instances_.size(); i++) {
        Instance *instance = task1_instances_[i];
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