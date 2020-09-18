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

#include <dynet/grad-check.h>
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

void SemanticPipe::Initialize() {
	Pipe::Initialize();
	PreprocessData();
	model_ = new ParameterCollection();
	pruner_model_ = new ParameterCollection();
	SemanticOptions *semantic_options = GetSemanticOptions();
	pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	if (semantic_options->trainer() == "adadelta") {
		trainer_ = new AdadeltaTrainer(*model_);
//		pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	} else if (semantic_options->trainer() == "adam") {
		trainer_ = new AdamTrainer(*model_, semantic_options->eta0());
//		pruner_trainer_ = new AdamTrainer(*pruner_model_,
//		                                  semantic_options->eta0());
	} else if (semantic_options->trainer() == "sgd") {
		trainer_ = new SimpleSGDTrainer(*model_, semantic_options->eta0());
//		pruner_trainer_ = new SimpleSGDTrainer(*pruner_model_,
//		                                       semantic_options->eta0());
	} else if (semantic_options->trainer() == "adagrad") {
		trainer_ = new AdagradTrainer(*model_, semantic_options->eta0());
//		pruner_trainer_ = new AdagradTrainer(*model_, semantic_options->eta0());
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}

	int num_roles = GetSemanticDictionary()->GetNumRoles();
	parser_ = new Parser(semantic_options, num_roles, GetSemanticDecoder(),
	                     model_);
	parser_->InitParams(model_);
	if (semantic_options->batch_size() == 1) {
		trainer_->clip_threshold = 1.0;
		pruner_trainer_->clip_threshold = 1.0;
	}
	pruner_ = new Pruner(semantic_options, num_roles, GetSemanticDecoder(),
	                     pruner_model_);
	pruner_->InitParams(pruner_model_);
}

void SemanticPipe::SaveModel(FILE *fs) {
	bool success;
	success = WriteUINT64(fs, kSemanticParserModelCheck);
	CHECK(success);
	success = WriteUINT64(fs, kSemanticParserModelVersion);
	CHECK(success);
	token_dictionary_->Save(fs);
	Pipe::SaveModel(fs);
	return;
}

void SemanticPipe::SaveNeuralModel() {
	string file_path = options_->GetModelFilePath() + ".dynet";
	save_dynet_model(file_path, model_);
//    TextFileSaver saver(file_path);
//    saver.save(*model_);
}

void SemanticPipe::SavePruner() {
	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path
			= semantic_options->GetPrunerModelFilePath();
//    TextFileSaver saver(file_path);
//    saver.save(*pruner_model_);
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
	Pipe::LoadModel(fs);
	return;
}

void SemanticPipe::LoadNeuralModel() {
	if (model_) delete model_;
	if (parser_) delete parser_;
	if (trainer_) delete trainer_;
	model_ = new ParameterCollection();
	SemanticOptions *semantic_options = GetSemanticOptions();

	int num_roles = GetSemanticDictionary()->GetNumRoles();
	if (semantic_options->trainer() == "adadelta") {
		trainer_ = new AdadeltaTrainer(*model_);
	} else if (semantic_options->trainer() == "adam") {
		trainer_ = new AdamTrainer(*model_, semantic_options->eta0());
	} else if (semantic_options->trainer() == "sgd") {
		trainer_ = new SimpleSGDTrainer(*model_, semantic_options->eta0());
	} else if (semantic_options->trainer() == "adagrad") {
		trainer_ = new AdagradTrainer(*model_, semantic_options->eta0());
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}
	trainer_->clip_threshold = 1.0;
	parser_ = new Parser(semantic_options, num_roles, GetSemanticDecoder(),
	                     model_);
	parser_->InitParams(model_);

	string file_path = options_->GetModelFilePath() + ".dynet";
	load_dynet_model(file_path, model_);

	model_->get_weight_decay().update_weight_decay(
			semantic_options->num_updates_);
}

void SemanticPipe::LoadPruner() {
	if (pruner_model_) delete pruner_model_;
	if (pruner_) delete pruner_;
	if (pruner_trainer_) delete pruner_trainer_;
	pruner_model_ = new ParameterCollection();
	SemanticOptions *semantic_options = GetSemanticOptions();
	const string file_path = semantic_options->GetPrunerModelFilePath();

	int num_roles = GetSemanticDictionary()->GetNumRoles();
	if (semantic_options->trainer() == "adadelta")
		pruner_trainer_ = new AdadeltaTrainer(*pruner_model_);
	else if (semantic_options->trainer() == "adam") {
		pruner_trainer_ = new AdamTrainer(*pruner_model_,
		                                  semantic_options->eta0());
	} else if (semantic_options->trainer() == "sgd") {
		pruner_trainer_ = new SimpleSGDTrainer(*model_,
		                                       semantic_options->eta0());
	} else if (semantic_options->trainer() == "adagrad") {
		trainer_ = new AdagradTrainer(*model_, semantic_options->eta0());
	} else {
		CHECK(false) << "Unsupported trainer. Giving up..." << endl;
	}
	pruner_trainer_->clip_threshold = 1.0;
	pruner_ = new Pruner(semantic_options, num_roles, GetSemanticDecoder(),
	                     pruner_model_);

	pruner_->InitParams(pruner_model_);
	load_dynet_model(file_path, pruner_model_);
	pruner_model_->get_weight_decay().update_weight_decay(
			semantic_options->pruner_num_updates_);
}

void SemanticPipe::PreprocessData() {
	delete token_dictionary_;
	CreateTokenDictionary();
	static_cast<SemanticDictionary *>(dictionary_)->SetTokenDictionary(
			token_dictionary_);
	static_cast<SemanticTokenDictionary *>(token_dictionary_)->Initialize(
			GetSemanticReader());
	static_cast<SemanticDictionary *>(dictionary_)->CreatePredicateRoleDictionaries(
			GetSemanticReader());
}

void SemanticPipe::MakeParts(Instance *instance, Parts *parts,
                             vector<double> *gold_outputs, int t) {
	SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
	semantic_parts->Initialize();
	bool make_gold = (gold_outputs != NULL);
	if (make_gold) gold_outputs->clear();
	parts->clear();
	if (GetSemanticOptions()->train_pruner()) {
		MakePartsBasic(instance, parts, gold_outputs, true, t);
		semantic_parts->BuildOffsets();
		semantic_parts->BuildIndices();
	} else {
		MakePartsLabeled(instance, parts, gold_outputs, t);
		semantic_parts->BuildOffsets();
		semantic_parts->BuildIndices();
	}
}

void SemanticPipe::MakePartsLabeled(Instance *instance, Parts *parts,
                                    vector<double> *gold_outputs, int t) {
	SemanticInstanceNumeric *sentence =
			static_cast<SemanticInstanceNumeric *>(instance);
	SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
	SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
	SemanticOptions *semantic_options = GetSemanticOptions();
	int slen = sentence->size() - 1;
	const int max_span_length = semantic_options->max_span_length();
	const int max_dist = semantic_options->max_dist();
	Parts *pruner_parts = CreateParts();
	vector<double> pruner_scores;
	vector<double> pruner_gold_outputs;
	vector<double> pruner_predicted_outputs;
	bool is_train = semantic_options->train();
	SemanticParts *pruner_semantic_parts
			= static_cast<SemanticParts *> (pruner_parts);

	if (semantic_options->prune_basic()) {
		MakePartsBasic(instance, pruner_parts, &pruner_gold_outputs, true, t);
		pruner_semantic_parts->BuildOffsets();
		pruner_semantic_parts->BuildIndices();
		ComputationGraph cg;
		pruner_->StartGraph(cg, false);
		Expression ex_loss = pruner_->BuildGraph(instance,
		                                         pruner_parts, &pruner_scores,
		                                         NULL,
		                                         &pruner_predicted_outputs,
		                                         form_count_, false, cg);
		double loss = as_scalar(cg.forward(ex_loss));
	}

	int op, np;
	static_cast<SemanticParts *> (pruner_parts)->GetOffsetPredicate(&op, &np);
	double threshold = GetSemanticOptions()->GetPrunerPosteriorThreshold() / (slen * slen * np);
	int den = slen * slen;

	int num_parts_initial = semantic_parts->size();
	int p_start, p_end;
	sentence->GetPredicateSpan(t, p_start, p_end);

	int lu_id = sentence->GetLu(t);
	int lu_name_id = sentence->GetLuName(t);
	int lu_pos_id = sentence->GetLuPos(t);

	vector<int> frames = semantic_dictionary->GetFramesByLu(lu_id);
	CHECK_GT(frames.size(), 0);

	for (auto frame_id: frames) {
		bool prune = false;
		vector<int> allowed_roles
				= semantic_dictionary->GetExistingRoles(frame_id);
		if (allowed_roles.size() == 0) continue;
		if (semantic_options->prune_basic() && !prune) {
			SemanticPartPredicate pruner_predicate(t, p_start, p_end,
			                                       lu_name_id, lu_pos_id,
			                                       frame_id);
			int pruner_r = pruner_semantic_parts->FindPredicate(
					pruner_predicate);
			CHECK_GE(pruner_r, 0);
			if (pruner_predicted_outputs[pruner_r] < threshold) prune = true;
		}

		PredicateNumeric predicate(p_start, p_end, lu_id, lu_name_id, lu_pos_id,
		                           frame_id);
		bool is_gold = sentence->FindPredicate(predicate) >= 0;
		if (prune && !(is_train && is_gold)) continue;

		Part *part = semantic_parts
				->CreatePartPredicate(t, p_start, p_end, lu_name_id, lu_pos_id,
				                      frame_id);
		semantic_parts->AddPredicatePart(part);
		if (is_gold) {
			gold_outputs->push_back(1.0);
		} else {
			gold_outputs->push_back(0.0);
		}
	}
	semantic_parts->SetOffsetPredicate(num_parts_initial,
	                                   semantic_parts->size() -
	                                   num_parts_initial);

	num_parts_initial = semantic_parts->size();
	int num_pred_parts = num_parts_initial;
//    LOG(INFO) << num_pred_parts;
	for (int p = 0; p < num_pred_parts; ++p) {
		SemanticPartPredicate *pred_part
				= static_cast<SemanticPartPredicate *>((*semantic_parts)[p]);
		int frame_id = pred_part->frame();
		vector<int> allowed_roles = semantic_dictionary->GetExistingRoles(
				frame_id);
		bool is_gold_pred = NEARLY_EQ_TOL((*gold_outputs)[p], 1.0, 1e-6);
		for (int a_start = 1; a_start < slen; ++a_start) {
			// donot include start or and in the parts
			for (int a_end = a_start; a_end < slen; ++a_end) {
				if (a_end - a_start > max_span_length) continue;
				bool prune = false;
				if (abs(a_start - p_start) > max_dist) prune = true;
				if (semantic_options->prune_basic() && !prune) {
					SemanticPartArgument pruner_argument(p, frame_id, a_start,
					                                     a_end, -1);
					int pruner_r = pruner_semantic_parts
							->FindArgument(frame_id, pruner_argument);
					CHECK_GE(pruner_r, 0);
					if (pruner_predicted_outputs[pruner_r] < threshold)
						prune = true;
				}

				CHECK_GT(allowed_roles.size(), 0);
				for (int r = 0; r < allowed_roles.size(); ++r) {
					int role = allowed_roles[r];
					bool is_gold = false;
					if (is_gold_pred) {
						ArgumentNumeric argument(a_start, a_end, role);
						is_gold = sentence->FindArgument(t, argument) >= 0;
					}
					if (prune && !(is_train && is_gold)) continue;
					// avoid pruning out gold parts during training
					Part *part = semantic_parts->CreatePartArgument(p, frame_id,
					                                                a_start,
					                                                a_end,
					                                                role);
					semantic_parts->AddArgumentPart(p, part);
					if (is_gold) {
						gold_outputs->push_back(1.0);
					} else {
						gold_outputs->push_back(0.0);
					}
				}
			}
		}
	}
	delete pruner_parts;
	semantic_parts->SetOffsetArgument(num_parts_initial,
	                                  semantic_parts->size() -
	                                  num_parts_initial);
}

void SemanticPipe::MakePartsBasic(Instance *instance, Parts *parts,
                                  vector<double> *gold_outputs, bool is_pruner,
                                  int t) {
	CHECK(is_pruner);
	SemanticInstanceNumeric *sentence =
			static_cast<SemanticInstanceNumeric *>(instance);
	SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
	SemanticDictionary *semantic_dictionary = GetSemanticDictionary();
	SemanticOptions *semantic_options = GetSemanticOptions();
	int slen = sentence->size() - 1;
	bool make_gold = (gold_outputs != NULL);
	if (make_gold) gold_outputs->clear();
	const int max_span_length = semantic_options->max_span_length();
	const int max_dist = semantic_options->max_dist();
	int num_parts_initial = semantic_parts->size();

	bool is_train = semantic_options->train();

	int p_start, p_end;
	sentence->GetPredicateSpan(t, p_start, p_end);
	int lu_id = sentence->GetLu(t);
	int lu_name_id = sentence->GetLuName(t);
	int lu_pos_id = sentence->GetLuPos(t);
	vector<int> frames = semantic_dictionary->GetFramesByLu(lu_id);
	bool gold_exists = false;
	for (auto frame_id: frames) {
		vector<int> allowed_roles
				= semantic_dictionary->GetExistingRoles(frame_id);
		if (allowed_roles.size() == 0) continue;
		PredicateNumeric predicate(p_start, p_end, lu_id, lu_name_id, lu_pos_id,
		                           frame_id);
		bool is_gold = sentence->FindPredicate(predicate) >= 0;
		Part *part = semantic_parts
				->CreatePartPredicate(t, p_start, p_end, lu_name_id, lu_pos_id,
				                      frame_id);
		semantic_parts->AddPredicatePart(part);
		if (is_gold) {
			gold_outputs->push_back(1.0);
			gold_exists = true;
		} else {
			gold_outputs->push_back(0.0);
		}
	}
	CHECK(gold_exists);
	semantic_parts->SetOffsetPredicate(num_parts_initial,
	                                   semantic_parts->size() -
	                                   num_parts_initial);

	num_parts_initial = semantic_parts->size();
	int num_pred_parts = num_parts_initial;
	for (int p = 0; p < num_pred_parts; ++p) {
		SemanticPartPredicate *pred_part
				= static_cast<SemanticPartPredicate *> ((*parts)[p]);
		int frame_id = pred_part->frame();
		vector<int> allowed_roles
				= semantic_dictionary->GetExistingRoles(frame_id);
		bool is_gold_pred = NEARLY_EQ_TOL((*gold_outputs)[p], 1.0, 1e-6);
		for (int a_start = 1; a_start < slen; ++a_start) {
			for (int a_end = a_start; a_end < slen; ++a_end) {
				if (a_end - a_start > max_span_length) continue;

				bool prune = false;
				if (abs(a_start - p_start) > max_dist) prune = true;
				bool is_gold = false;
				if (is_gold_pred) {
					for (auto role: allowed_roles) {
						ArgumentNumeric argument(a_start, a_end, role);
						int l = sentence->FindArgument(t, argument);
						if (l >= 0) {
							is_gold = true;
							break;
						}
					}
				}
				if (prune && !(is_train && is_gold)) continue;
				Part *part = semantic_parts->CreatePartArgument(p, frame_id,
				                                                a_start, a_end,
				                                                -1);
				semantic_parts->AddArgumentPart(p, part);
				if (is_gold) {
					gold_outputs->push_back(1.0);
				} else {
					gold_outputs->push_back(0.0);
				}
			}
		}
	}
	semantic_parts->SetOffsetUnlabeledArgument(num_parts_initial,
	                                           semantic_parts->size() -
	                                           num_parts_initial);
	return;
}

void SemanticPipe::LabelInstance(Parts *parts,
                                 const vector<double> &gold_output,
                                 const vector<double> &predicted_output,
                                 Instance *gold_instance,
                                 Instance *predicted_instance) {
	SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
	SemanticInstance *gold_semantic_instance =
			static_cast<SemanticInstance *>(gold_instance);
	SemanticInstance *predicted_semantic_instance =
			static_cast<SemanticInstance *>(predicted_instance);
	SemanticDictionary *semantic_dictionary =
			static_cast<SemanticDictionary *>(dictionary_);

	int slen = gold_semantic_instance->size();
	double threshold = 0.5;
	int offset_pred_parts, num_pred_parts;
	semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
	int offset_arg_parts, num_arg_parts;
	semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);

	for (int p = 0; p < num_pred_parts; ++p) {
//        int s, int e, string name, string frame
		Part *p_part = (*semantic_parts)[offset_pred_parts + p];
		SemanticPartPredicate *pred_part =
				static_cast<SemanticPartPredicate *> (p_part);
		int p_r = semantic_parts->FindPredicate(*pred_part);
		CHECK_GE(p_r, 0);
		CHECK_LT(p_r, parts->size());

		if (predicted_output[p_r] < threshold &&
		    gold_output[p_r] < threshold)
			continue;

		if (NEARLY_EQ_TOL(predicted_output[p_r], 1, 1e-6)) npa_ += 1;
		if (NEARLY_EQ_TOL(gold_output[p_r], 1, 1e-6)) {
			nga_ += 1;
			num_gold_frames_ += 1;
		}
		if (NEARLY_EQ_TOL(predicted_output[p_r], gold_output[p_r], 1e-6)) {
			nma_ += 1;
			num_matched_frames_ += 1;
		}

		int ps, pe;
		pred_part->span(ps, pe);
		string lu_name = semantic_dictionary->GetLuName(pred_part->lu_name());
		string lu_pos = semantic_dictionary->GetLuPos(pred_part->lu_pos());
		string frame = semantic_dictionary->GetFrame(pred_part->frame());
		string lu = lu_name + "." + lu_pos;
		// int s, int e, string name, string frame
		Predicate predicate(ps, pe, lu, frame);
		vector<Argument> predicted_arguments;
		vector<Argument> gold_arguments;
		vector<int> sequence_labels(slen, -1);

		vector<int> arguments_by_predicate = semantic_parts->GetArgumentsByPredicate(
				p);
		for (auto a_r: arguments_by_predicate) {
			CHECK_GE(a_r, 0);
			CHECK_LT(a_r, parts->size());
			Part *a_part = (*semantic_parts)[a_r];
			SemanticPartArgument *arg_part = static_cast<SemanticPartArgument *> (a_part);
			int as, ae;
			arg_part->span(as, ae);


//            LOG(INFO) << a_r <<" " <<as<< " " << ae  <<" "
//                      <<arg_part->pred_idx() <<" " << predicted_output[a_r] <<" "<<gold_output[a_r]<<endl;

			if (predicted_output[a_r] > threshold) {
				CHECK(NEARLY_EQ_TOL(predicted_output[a_r], 1.0, 1e-6));
				for (int i = as; i <= ae; ++i) {
					CHECK_EQ(sequence_labels[i], -1);
					sequence_labels[i] = arg_part->role();
				}
			}
			if (gold_output[a_r] > threshold) {
				CHECK_EQ(gold_output[a_r], 1.0);
				string role = semantic_dictionary->GetRoleName(
						arg_part->role());
				gold_arguments.push_back(Argument(as, ae, role));
			}
		}

		int as = -1, ae = -1, prev_role = -1, curr_role = -1;
		for (int i = 0; i < slen; ++i) {
			curr_role = sequence_labels[i];
			if (curr_role != prev_role) {
				if (prev_role != -1) {
					CHECK_GE(as, 1);
					ae = i - 1;
					CHECK_GE(ae, as);
					string role = semantic_dictionary->GetRoleName(prev_role);
					predicted_arguments.push_back(Argument(as, ae, role));
					ae = -1;
					if (curr_role == -1) as = -1;
					else as = i;
				} else {
					CHECK_EQ(as, -1);
					as = i;
				}
			}
			prev_role = curr_role;
		}


		if (predicted_output[p_r] > threshold) {
			CHECK_EQ(predicted_output[p_r], 1.0);
			predicted_semantic_instance->AddPredicate(predicate,
			                                          predicted_arguments);
		}
		if (gold_output[p_r] > threshold) {
			CHECK_EQ(gold_output[p_r], 1.0);
			gold_semantic_instance->AddPredicate(predicate, gold_arguments);
		}
	}
}

void SemanticPipe::EvaluateInstance(Instance *instance, Instance *gold_instance,
                                    Instance *predicted_instance) {
	int num_possible_arguments = 0;
	int num_gold_arguments = 0;

	SemanticInstance *semantic_instance =
			static_cast<SemanticInstance *>(instance);
	SemanticInstance *gold_semantic_instance =
			static_cast<SemanticInstance *>(gold_instance);
	SemanticInstance *predicted_semantic_instance =
			static_cast<SemanticInstance *>(predicted_instance);

	SemanticDictionary *semantic_dictionary =
			static_cast<SemanticDictionary *>(dictionary_);

	int slen = semantic_instance->size();
	for (int p = 0; p < gold_semantic_instance->GetNumPredicates(); ++p) {
		num_gold_arguments += gold_semantic_instance->GetNumArgumentsPredicate(
				p);
	}

	int num_actual_gold_arguments = 0;
	int as, ae;
	for (int p = 0; p < semantic_instance->GetNumPredicates(); ++p) {
		int num_gold_args = semantic_instance->GetNumArgumentsPredicate(p);
		num_actual_gold_arguments += num_gold_args;
		int num_predicted_arguments = predicted_semantic_instance->GetNumArgumentsPredicate(
				p);
		num_predicted_arguments_ += num_predicted_arguments;
		int frame_id = semantic_dictionary->GetFrameAlphabet().Lookup(
				semantic_instance->GetPredicateFrame(p));
		CHECK_GE(frame_id, 0);
		for (int ga = 0; ga < num_gold_args; ++ga) {
			Argument gold_argument = semantic_instance->GetArgument(p, ga);
			int gold_role_id = semantic_dictionary->GetRoleAlphabet().Lookup(
					gold_argument.GetRole());
			CHECK_GE(gold_role_id, 0);
			double score = 0.5;
			if (semantic_dictionary->IsCoreRole(frame_id, gold_role_id))
				score = 1.0;
			for (int pa = 0; pa < num_predicted_arguments; ++pa) {
				if (predicted_semantic_instance->GetArgument(p, pa) ==
				    gold_argument) {
					++num_matched_arguments_;
					nma_ += score;
					break;
				}
			}
			nga_ += score;
		}

		for (int pa = 0; pa < num_predicted_arguments; ++pa) {
			Argument predicted_argument = predicted_semantic_instance->GetArgument(
					p, pa);
			int predicted_role_id = semantic_dictionary->GetRoleAlphabet().Lookup(
					predicted_argument.GetRole());
			CHECK_GE(predicted_role_id, 0);
			double score = 0.5;
			if (semantic_dictionary->IsCoreRole(frame_id, predicted_role_id))
				score = 1.0;
			npa_ += score;
		}
	}
	int missed_gold_arguments = num_actual_gold_arguments - num_gold_arguments;
	num_pruned_gold_arguments_ += missed_gold_arguments;
	num_gold_arguments_ += num_actual_gold_arguments;
}

void SemanticPipe::Train() {

	CreateInstances();
	SaveModelFile();
	SemanticOptions *semantic_options = GetSemanticOptions();
	//LoadPruner(semantic_options->GetPrunerModelFilePath());
	if (semantic_options->use_pretrained_embedding()) {
		LoadPretrainedEmbedding(true, false);
	}
	BuildFormCount();
	//CHECK(semantic_options->prune_basic());
	if (semantic_options->prune_basic()) LoadPruner();

	vector<int> idxs, exemplar_idxs;
	for (int i = 0; i < instances_.size(); ++i)
		idxs.push_back(i);
	for (int i = 0; i < exemplar_instances_.size(); ++i)
		exemplar_idxs.push_back(i);
	double F1 = 0, best_F1 = -1;

//    semantic_options->train_off();
//    Run(F1);
//    return;
	for (int i = 0; i < options_->GetNumEpochs(); ++i) {
		semantic_options->train_on();
		random_shuffle(idxs.begin(), idxs.end());
		random_shuffle(exemplar_idxs.begin(), exemplar_idxs.end());
		TrainEpoch(idxs, exemplar_idxs, i, best_F1);
		semantic_options->train_off();
		Run(F1);
		if (F1 > best_F1) {
			SaveModelFile();
			SaveNeuralModel();
			best_F1 = F1;
		}
	}
}

void SemanticPipe::TrainPruner() {
	CreateInstances();
	SaveModelFile();
	SemanticOptions *semantic_options = GetSemanticOptions();

	BuildFormCount();
	vector<int> idxs;
	for (int i = 0; i < instances_.size(); ++i) {
		idxs.push_back(i);
	}
	double acc = 0, best_acc = -1;
	for (int i = 0; i < 5; ++i) {
		semantic_options->train_on();
		random_shuffle(idxs.begin(), idxs.end());
		TrainPrunerEpoch(idxs, i);
		semantic_options->train_off();
		RunPruner(acc);
		LOG(INFO) << semantic_options->pruner_num_updates_;
		SaveModelFile();
		SavePruner();
	}
}

double SemanticPipe::TrainEpoch(const vector<int> &idxs,
                                const vector<int> &exemplar_idxs,
                                int epoch, double &best_F1) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, NULL);
	vector<Parts *> parts(batch_size, NULL);
	for (int i = 0;i < batch_size; ++ i) parts[i] = CreateParts();
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());
	vector<string> splits(batch_size);
	vector<int> ids(batch_size);

	float forward_loss = 0.0;
	int num_instances = idxs.size();
	int num_exemplar_instances =
			exemplar_idxs.size() * semantic_options->exemplar_fraction();

	if (epoch == 0) {
		LOG(INFO) << "Number of instances: " << num_instances
		          << "; Number of exemplars: " << num_exemplar_instances
		          << endl;
	}
	LOG(INFO) << " Iteration #" << epoch + 1;

	int ite = 0, exemplar_ite = 0;
	int checkpoint_ite = 0;

	for (int i = 0; i < num_instances + num_exemplar_instances; i += batch_size) {
		int n_batch = min(batch_size, num_instances + num_exemplar_instances - i);
		for (int j = 0; j < n_batch; ++j) {
			float rand_float =
					static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			bool choose_exemplar = rand_float < num_exemplar_instances * 1.0 /
			                                    (num_exemplar_instances +
			                                     num_instances);
			if (exemplar_ite >= num_exemplar_instances) choose_exemplar = false;
			if (ite >= num_instances) choose_exemplar = true;

			if (choose_exemplar) {
				splits[j] = "exemplar";
				ids[j] = exemplar_idxs[exemplar_ite];
				instance[j] = exemplar_instances_[exemplar_idxs[exemplar_ite++]];
			} else {
				splits[j] = "train";
				ids[j] = idxs[ite];
				instance[j] = instances_[idxs[ite++]];
			}
			MakeParts(instance[j], parts[j], &(gold_outputs[j]), 0);
		}
		ComputationGraph *cg = new ComputationGraph;
		parser_->StartGraph(*cg, true);
		vector <Expression> ex_losses;
		for (int j = 0; j < n_batch; ++j) {
			Expression i_loss
					= parser_->BuildGraph(instance[j], parts[j], &scores[j],
					                      &gold_outputs[j], &predicted_outputs[j],
					                      form_count_, splits[j], ids[j], true, *cg);
			ex_losses.push_back(i_loss);
		}
		Expression ex_loss = sum(ex_losses);
		double loss = max(float(0.0), as_scalar(cg->forward(ex_loss)));
		forward_loss += loss;
		int corr = 0, num_parts = 0;
		for (int j = 0; j < n_batch; ++j) {
			for (int r = 0; r < parts[j]->size(); ++r) {
				if (NEARLY_EQ_TOL(gold_outputs[j][r], predicted_outputs[j][r], 1e-6))
					corr += 1;
				else break;
			}
			num_parts += parts[j]->size();
		}
		if (corr < num_parts) {
			cg->backward(ex_loss);
			trainer_->update();
			++semantic_options->num_updates_;
		}
		delete cg;
		checkpoint_ite += n_batch;
		if (checkpoint_ite > 10000 && epoch >= 20) {
			double F1 = 0;
			semantic_options->train_off();
			Run(F1);
			if (F1 > best_F1 && F1 > 0.7) {
				SaveModelFile();
				SaveNeuralModel();
				best_F1 = F1;
			}
			semantic_options->train_on();
			checkpoint_ite = 0;
		}
	}
	if (semantic_options->halve() > 0 &&
	    (epoch + 1) % semantic_options->halve() == 0) {
		semantic_options->eta0_ /= 2;
		trainer_->learning_rate /= 2;
	}
	if (semantic_options->trainer() == "sgd") {
		trainer_->learning_rate
				= semantic_options->eta0() / (1 + (epoch + 1) * semantic_options->eta_decay());
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	parts.clear();
	LOG(INFO) << "training loss: " << forward_loss / (num_instances + num_exemplar_instances) << endl;
	trainer_->status();
	return forward_loss;
}

double SemanticPipe::TrainPrunerEpoch(const vector<int> &idxs, int epoch) {
	SemanticOptions *semantic_options = GetSemanticOptions();
	int batch_size = semantic_options->batch_size();
	vector<Instance *> instance(batch_size, NULL);
	vector<Parts *> parts(batch_size, NULL);
	for (int i = 0;i < batch_size; ++ i) parts[i] = CreateParts();
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());

	double forward_loss = 0.0;
	int num_instances = idxs.size();
	if (epoch == 0) {
		LOG(INFO) << "Number of instances: " << num_instances << endl;
	}
	LOG(INFO) << " Iteration #" << epoch + 1;
	for (int i = 0; i < num_instances; i += batch_size) {
		int n_batch = min(batch_size, num_instances - i);
		for (int j = 0; j < n_batch; ++j) {
			instance[j] = instances_[idxs[i + j]];
			MakeParts(instance[j], parts[j], &gold_outputs[j], 0);
		}
		ComputationGraph cg;
		pruner_->StartGraph(cg, true);
		vector <Expression> ex_losses;
		for (int j = 0; j < n_batch; ++j) {
			Expression i_loss
					= pruner_->BuildGraph(instance[j], parts[j], &scores[j],
					                      &gold_outputs[j], &predicted_outputs[j],
					                      form_count_, true, cg);
			ex_losses.push_back(i_loss);
		}
		Expression ex_loss = sum(ex_losses);

		double loss = as_scalar(cg.forward(ex_loss));
		cg.backward(ex_loss);
		pruner_trainer_->update();
		++semantic_options->pruner_num_updates_;
		forward_loss += loss;
	}

	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	parts.clear();
	LOG(INFO) << "training loss: " << forward_loss / num_instances << endl;
	return forward_loss;
}

void SemanticPipe::Test() {
	SemanticOptions *semantic_options = GetSemanticOptions();
	LoadNeuralModel();
	LoadPruner();
	semantic_options->train_off();
	double F1 = 0;
	Run(F1);
}

void SemanticPipe::Run(double &F1) {
	int batch_size = GetSemanticOptions()->batch_size();
//	batch_size = 1;
	vector<Instance *> instance(batch_size, NULL);
	vector<Parts *> parts(batch_size, NULL);
	for (int i = 0;i < batch_size; ++ i) parts[i] = CreateParts();
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());

	timeval start, end;
	gettimeofday(&start, NULL);

	string test_file = GetSemanticOptions()->GetTestFilePath();
	vector<string> fields;
	StringSplit(test_file, "/", &fields, true);
	string split = fields[fields.size() - 1];

	if (options_->evaluate()) BeginEvaluation();
	writer_->Open(options_->GetOutputFilePath());
	int num_instances = dev_instances_.size();
	double forward_loss = 0.0;
	for (int i = 0; i < num_instances; i += batch_size) {
		int n_batch = min(batch_size, num_instances - i);
		for (int j = 0;j < n_batch; ++ j) {
			instance[j] = GetFormattedInstance(dev_instances_[i + j]);
			MakeParts(instance[j], parts[j], &gold_outputs[j], 0);
		}
		ComputationGraph cg;
		parser_->StartGraph(cg, false);
		vector<Expression> ex_losses;
		for (int j = 0;j < n_batch; ++ j) {
			Expression i_loss = parser_->BuildGraph(instance[j],
			                                        parts[j], &scores[j],
			                                        &gold_outputs[j],
			                                        &predicted_outputs[j],
			                                        form_count_, split, i + j, false, cg);
			ex_losses.push_back(i_loss);
//			predicted_outputs[j].assign(parts[j]->size(), 0.0);
		}
		Expression ex_loss = sum(ex_losses);
		double loss = max(float(0.0), as_scalar(cg.forward(ex_loss)));
		forward_loss += loss;

		for (int j = 0; j < n_batch; ++j) {
			Instance *predicted_instance = dev_instances_[i + j]->Copy();
			Instance *gold_instance = dev_instances_[i + j]->Copy();
			static_cast<SemanticInstance *> (predicted_instance)->ClearPredicates();
			static_cast<SemanticInstance *> (gold_instance)->ClearPredicates();
			LabelInstance(parts[j], gold_outputs[j], predicted_outputs[j],
			              gold_instance, predicted_instance);
			if (options_->evaluate()) {
				EvaluateInstance(dev_instances_[i + j], gold_instance, predicted_instance);
			}
			static_cast<SemanticWriter *> (writer_)->Write(predicted_instance);
			if (instance[j] != dev_instances_[i + j]) delete instance[j];
			delete predicted_instance;
			delete gold_instance;
		}
	}
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}
	LOG(INFO) << "dev loss: " << forward_loss / num_instances << endl;
	writer_->Close();
	gettimeofday(&end, NULL);

	if (options_->evaluate()) EndEvaluation(F1);
}

void SemanticPipe::RunPruner(double &accuracy) {
	int batch_size = GetSemanticOptions()->batch_size();
//	batch_size = 1;
	vector<Instance *> instance(batch_size, NULL);
	vector<Parts *> parts(batch_size, NULL);
	for (int i = 0;i < batch_size; ++ i) parts[i] = CreateParts();
	vector<vector<double>> scores(batch_size, vector<double> ());
	vector<vector<double>> gold_outputs(batch_size, vector<double> ());
	vector<vector<double>> predicted_outputs(batch_size, vector<double> ());

	timeval start, end;
	gettimeofday(&start, NULL);

	int num_instances = dev_instances_.size();
	if (options_->evaluate()) BeginEvaluation();
	double forward_loss = 0.0;
	int corr = 0, total = 0, pruned = 0, gold = 0;
	double th = GetSemanticOptions()->GetPrunerPosteriorThreshold();
	int total_slen = 0;
	for (int i = 0; i < num_instances; i += batch_size) {
//		LOG(INFO) << i <<"/" <<num_instances;
		int n_batch = min(batch_size, num_instances - i);
		for (int j = 0;j < n_batch; ++ j) {
			instance[j] = GetFormattedInstance(dev_instances_[i + j]);
			MakeParts(instance[j], parts[j], &gold_outputs[j], 0);
		}
		ComputationGraph cg;
		pruner_->StartGraph(cg, false);
		vector<Expression> ex_losses;
		for (int j = 0;j < n_batch; ++ j) {
			Expression i_loss = pruner_->BuildGraph(instance[j],
			                                        parts[j], &scores[j],
			                                        &gold_outputs[j], &predicted_outputs[j],
			                                        form_count_, false, cg);
			ex_losses.push_back(i_loss);
		}
		Expression ex_loss = sum(ex_losses);
		double loss = as_scalar(cg.forward(ex_loss));
		forward_loss += loss;
		for (int j = 0;j < n_batch; ++ j) {
			SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts[j]);
			int offset_arg_parts, num_arg_parts;
			semantic_parts->GetOffsetUnlabeledArgument(&offset_arg_parts,
			                                           &num_arg_parts);
			int offset_pred_parts, num_pred_parts;
			semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
			int slen = static_cast<SemanticInstanceNumeric *>(instance[j])->size() - 1;
			total_slen += slen;
			double threshold = th / (slen * slen * num_pred_parts);
			for (int k = 0; k < num_arg_parts; ++k) {
				int r = k + offset_arg_parts;
				CHECK_EQ((*parts[j])[r]->type(), SEMANTICPART_ARGUMENT);
				if (gold_outputs[j][r] > 0.5) {
					++gold;
					if (predicted_outputs[j][r] > threshold) ++corr;
				}
				if (!(predicted_outputs[j][r] > threshold)) ++pruned;
				++total;
			}
			delete instance[j];
		}
	}
	forward_loss /= num_instances;
	accuracy = corr * 1.0 / gold;
	LOG(INFO) << "pruning dev loss: " << forward_loss << ". accuracy: "
	          << accuracy << " pruned: " << pruned << "/"
	          << total << endl;
	LOG(INFO) << (total - pruned) * 1.0 / total_slen << endl;
	for (int i = 0;i < batch_size; ++ i) {
		if (parts[i]) delete parts[i];
	}

	parts.clear();
	gettimeofday(&end, NULL);
}

void SemanticPipe::LoadPretrainedEmbedding(bool load_parser_embedding,
                                           bool load_pruner_embedding) {
	SemanticOptions *semantic_option = GetSemanticOptions();
	embedding_ = new unordered_map<int, vector<float>>();
	unsigned dim = semantic_option->word_dim();
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
		int form_id = token_dictionary_->GetFormId(word);
		if (form_id < 0)
			continue;
		found += 1;
		(*embedding_)[form_id] = v;
	}
	in.close();
	LOG(INFO) << found << "/" << token_dictionary_->GetNumForms()
	          << " words found in the pretrained embedding" << endl;
	if (load_parser_embedding) parser_->LoadEmbedding(embedding_);
	if (load_pruner_embedding) pruner_->LoadEmbedding(embedding_);
	delete embedding_;
}

void SemanticPipe::BuildFormCount() {
	bool use_exemplar = GetSemanticOptions()->use_exemplar();
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

	if (use_exemplar) {
		for (int i = 0; i < exemplar_instances_.size(); i++) {
			Instance *instance = exemplar_instances_[i];
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
}