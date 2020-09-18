//
// Created by hpeng on 7/9/17.
//

#include "Parser.h"

void Parser::InitParams(ParameterCollection *model) {
	// shared
	lookup_params_ = {
			{"embed_word_",   model->add_lookup_parameters(VOCAB_SIZE,
			                                               {WORD_DIM})},
			{"embed_lemma_",  model->add_lookup_parameters(LEMMA_SIZE,
			                                               {LEMMA_DIM})},
			{"embed_pos_",    model->add_lookup_parameters(POS_SIZE,
			                                               {POS_DIM})},
			{"embed_pred_length_", model->add_lookup_parameters(5, {4})},
			{"embed_lu_",     model->add_lookup_parameters(LU_SIZE, {LU_DIM})},
			{"embed_lu_pos_", model->add_lookup_parameters(LU_SIZE, {POS_DIM})},
			{"embed_frame_",  model->add_lookup_parameters(FRAME_SIZE,
			                                               {FRAME_DIM})},
			{"embed_role_",   model->add_lookup_parameters(ROLE_SIZE,
			                                               {ROLE_DIM})}
	};

	params_ = {
			{"pred_w_len_", model->add_parameters({MLP_DIM, 1})},
			{"pred_w_lstm_",       model->add_parameters({MLP_DIM, 4 * LSTM_DIM})},
			{"pred_b_lstm_",       model->add_parameters({MLP_DIM})},
			{"pred_w_span_",       model->add_parameters({MLP_DIM, MLP_DIM})},
			{"pred_b_span_",       model->add_parameters({MLP_DIM})},
			{"pred_tensor_span_",  model->add_parameters({RANK, MLP_DIM})},
			{"pred_w_lu_",         model->add_parameters({MLP_DIM, LU_DIM + POS_DIM})},
			{"pred_b_lu_",         model->add_parameters({MLP_DIM})},
			{"pred_tensor_lu_",    model->add_parameters({RANK, MLP_DIM})},
			{"pred_w_frame_",      model->add_parameters({MLP_DIM, FRAME_DIM})},
			{"pred_b_frame_",      model->add_parameters({MLP_DIM})},
			{"pred_tensor_frame_", model->add_parameters({RANK, MLP_DIM})},
			{"pred_out_b_",        model->add_parameters({FRAME_SIZE})},

			{"arg_w_len_",         model->add_parameters({MLP_DIM, 1})},
			{"arg_w_dist_",        model->add_parameters({MLP_DIM, 1})},
			{"arg_w_start_",       model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"arg_w_end_",         model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"arg_b_lstm_",        model->add_parameters({MLP_DIM})},
			{"arg_w_span_",        model->add_parameters({MLP_DIM, MLP_DIM})},
			{"arg_b_span_",        model->add_parameters({MLP_DIM})},
			{"arg_tensor_span_",   model->add_parameters({RANK, MLP_DIM})},
			{"arg_w_role_",        model->add_parameters({MLP_DIM, ROLE_DIM})},
			{"arg_b_role_",        model->add_parameters({MLP_DIM})},
			{"arg_tensor_role_",   model->add_parameters({RANK, MLP_DIM})},
			{"arg_out_b_",         model->add_parameters({ROLE_SIZE})}
	};
	if (USE_ELMO) {
		params_["elmo_w_in_"] = model->add_parameters({LSTM_DIM, ELMO_DIM + WORD_DIM + POS_DIM + LEMMA_DIM});
		params_["elmo_b_in_"] = model->add_parameters({LSTM_DIM});
		params_["elmo_att_"] = model->add_parameters({1, 3});
	}
}

void Parser::StartGraph(ComputationGraph &cg, bool is_train) {
	cg_params_.clear();
	if (DROPOUT > 0 && is_train) {
		l2rbuilder_.set_dropout(DROPOUT);
		r2lbuilder_.set_dropout(DROPOUT);
	} else {
		l2rbuilder_.disable_dropout();
		r2lbuilder_.disable_dropout();
	}
	l2rbuilder_.new_graph(cg);
	r2lbuilder_.new_graph(cg);
	for (auto it = params_.begin(); it != params_.end(); ++it) {
		cg_params_[it->first] = parameter(cg, it->second);
	}
}

void Parser::ReadWord(Instance *instance,
                      vector<Expression> &ex_words,
                      unordered_map<int, int> *form_count,
                      const string &split, int instance_id,
                      bool is_train, ComputationGraph &cg) {
	auto sentence = static_cast<SemanticInstanceNumeric *>(instance);
	const int slen = sentence->size();
	const vector<int> words = sentence->GetFormIds();
	const vector<int> lemmas = sentence->GetLemmaIds();
	const vector<int> pos = sentence->GetPosIds();

	vector<Expression> ex_elmos(slen);
	Expression elmo_w_in, elmo_b_in;
	if (USE_ELMO) {
		elmo_w_in = cg_params_.at("elmo_w_in_");
		elmo_b_in = cg_params_.at("elmo_b_in_");
		ReadELMo(split, instance_id, slen - 2, ex_elmos, cg); // remove start/end
	}
	for (int i = 0; i < slen; ++i) {
		int word_idx = words[i];
		int lemma_idx = lemmas[i];
		if (is_train && WORD_DROPOUT > 0.0 && word_idx != UNK_ID) {
			int count = form_count->at(word_idx);
			float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
			if (rand_float < WORD_DROPOUT / (static_cast<float> (count) + WORD_DROPOUT)) {
				word_idx = UNK_ID;
				lemma_idx = UNK_ID;
			}
		}
		Expression x_word = lookup(cg, lookup_params_.at("embed_word_"), word_idx);
		Expression x_lemma = lookup(cg, lookup_params_.at("embed_lemma_"), lemma_idx);
		Expression x_pos = lookup(cg,lookup_params_.at("embed_pos_"), pos[i]);

		if (USE_ELMO && i > 0 && i < slen - 1) {
			Expression x_elmo = ex_elmos[i - 1];
			ex_words[i] = affine_transform({elmo_b_in, elmo_w_in,
			                                concatenate({x_elmo, x_word, x_lemma, x_pos})});
		} else {
			ex_words[i] = concatenate({x_word, x_lemma, x_pos});
		}

		if (is_train && DROPOUT > 0) {
			ex_words[i] = dropout(ex_words[i], DROPOUT);
		}
	}
}


void Parser::RunLSTM(Instance * instance, LSTMBuilder &l2rbuilder, LSTMBuilder &r2lbuilder,
                     const vector<Expression> &ex_words, vector<Expression> &ex_lstm,
                     bool is_train, ComputationGraph &cg) {
	SemanticInstanceNumeric *sentence =
			static_cast<SemanticInstanceNumeric *>(instance);
	const int slen = sentence->size();

	l2rbuilder.start_new_sequence();
	r2lbuilder.start_new_sequence();

	vector<Expression> ex_l2r(slen), ex_r2l(slen);
	ex_lstm.resize(slen);
	for (int i = 0; i < slen; ++i) {
		ex_l2r[i] = l2rbuilder.add_input(ex_words[i]);
	}
	for (int i = 0; i < slen; ++i) {
		ex_r2l[slen - i - 1] = r2lbuilder.add_input(ex_words[slen - i - 1]);
		ex_lstm[slen - i - 1] = concatenate({ex_l2r[slen - i - 1], ex_r2l[slen - i - 1]});
	}
}

Expression Parser::BuildGraph(Instance *instance,
                              Parts *parts, vector<double> *scores,
                              const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                              unordered_map<int, int> *form_count, const string &split, int instance_id,
                              bool is_train, ComputationGraph &cg) {
	l2rbuilder_.start_new_sequence();
	r2lbuilder_.start_new_sequence();

	SemanticInstanceNumeric *sentence =
			static_cast<SemanticInstanceNumeric *>(instance);

	SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);

	int offset_pred_parts, num_pred_parts;
	semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
	int offset_arg_parts, num_arg_parts;
	semantic_parts->GetOffsetArgument(&offset_arg_parts, &num_arg_parts);
	const int slen = sentence->size();

	/* FN */
	Expression pred_w_len = cg_params_.at("pred_w_len_");
	Expression pred_w_lstm = cg_params_.at("pred_w_lstm_");
	Expression pred_b_lstm = cg_params_.at("pred_b_lstm_");
	Expression pred_w_span = cg_params_.at("pred_w_span_");
	Expression pred_b_span = cg_params_.at("pred_b_span_");
	Expression pred_tensor_span = cg_params_.at("pred_tensor_span_");

	Expression pred_w_lu = cg_params_.at("pred_w_lu_");
	Expression pred_b_lu = cg_params_.at("pred_b_lu_");
	Expression pred_tensor_lu = cg_params_.at("pred_tensor_lu_");
	Expression pred_w_frame = cg_params_.at("pred_w_frame_");
	Expression pred_b_frame = cg_params_.at("pred_b_frame_");
	Expression pred_tensor_frame = cg_params_.at("pred_tensor_frame_");
	Expression pred_out_b = cg_params_.at("pred_out_b_");

	Expression arg_w_len = cg_params_.at("arg_w_len_");
	Expression arg_w_dist = cg_params_.at("arg_w_dist_");
	Expression arg_w_start = cg_params_.at("arg_w_start_");
	Expression arg_w_end = cg_params_.at("arg_w_end_");
	Expression arg_b_lstm = cg_params_.at("arg_b_lstm_");
	Expression arg_w_span = cg_params_.at("arg_w_span_");
	Expression arg_b_span = cg_params_.at("arg_b_span_");
	Expression arg_tensor_span = cg_params_.at("arg_tensor_span_");
	Expression arg_w_role = cg_params_.at("arg_w_role_");
	Expression arg_b_role = cg_params_.at("arg_b_role_");
	Expression arg_tensor_role = cg_params_.at("arg_tensor_role_");
	Expression arg_out_b = cg_params_.at("arg_out_b_");

	vector<Expression> ex_words(slen), ex_lstm(slen);

	ReadWord(instance, ex_words, form_count, split, instance_id, is_train, cg);
	RunLSTM(instance, l2rbuilder_, r2lbuilder_, ex_words, ex_lstm, is_train, cg);

	vector<Expression> ex_arg_start(slen), ex_arg_end(slen);
	for (int i = 0; i < slen; ++i) {
		ex_arg_start[i] = arg_w_start * ex_lstm[i];
		ex_arg_end[i] = arg_w_end * ex_lstm[i];
	}

	vector<Expression> ex_preds(num_pred_parts);
	vector<Expression> ex_scores(parts->size());
	scores->assign(parts->size(), 0.0);
	predicted_outputs->assign(parts->size(), 0.0);

	int p_start, p_end, a_start, a_end;
	auto pred_part = static_cast<SemanticPartPredicate *> ((*parts)[offset_pred_parts]);
	pred_part->span(p_start, p_end);

	{
		float binned_len = Bin(p_end - p_start + 1, false);
		Expression ex_len = pred_w_len * binned_len;
		Expression ex_i = tanh(affine_transform({pred_b_lstm, pred_w_lstm,
		                                         concatenate({ex_lstm[p_start],
		                                                      ex_lstm[p_end]})}));
		Expression ex_h = tanh(
				affine_transform({pred_b_span, pred_w_span, ex_i}) + ex_len);
		Expression ex_pred_span = pred_tensor_span * ex_h;

		CHECK_EQ(0, offset_pred_parts);
		for (int p = 0; p < num_pred_parts; ++p) {
			auto pred_part = static_cast<SemanticPartPredicate *> ((*semantic_parts)[p]);
			int lu_name_id = pred_part->lu_name();
			int lu_pos_id = pred_part->lu_pos();
			int frame_id = pred_part->frame();
			Expression ex_lu = concatenate(
					{lookup(cg, lookup_params_.at("embed_lu_"), lu_name_id),
					 lookup(cg, lookup_params_.at("embed_lu_pos_"),
					        lu_pos_id)});
			Expression ex_pred_lu = pred_tensor_lu
			                        * affine_transform({pred_b_lu, pred_w_lu, ex_lu});
			Expression ex_pred_frame = pred_tensor_frame
			                           * affine_transform(
					{pred_b_frame, pred_w_frame,
					 lookup(cg, lookup_params_.at("embed_frame_"), frame_id)});
			ex_preds[p] = cmult(ex_pred_span,
			                    cmult(ex_pred_lu, ex_pred_frame));
			ex_scores[p] = pick(pred_out_b, frame_id) + sum_rows(ex_preds[p]);
			(*scores)[p] = as_scalar(cg.incremental_forward(ex_scores[p]));
		}
	}

	unordered_map<int, Expression> ex_span_cache;
	unordered_map<int, Expression> ex_role_cache;
	for (int i = 0; i < num_arg_parts; ++i) {
		int r = i + offset_arg_parts;
		auto arg = static_cast<SemanticPartArgument *>((*parts)[r]);
		int role = arg->role();
		int pred_r = arg->pred_idx();
		arg->span(a_start, a_end);
		if (ex_span_cache.find(a_start * slen + a_end) == ex_span_cache.end()) {
			ex_span_cache[a_start * slen + a_end]
					= tanh(ex_arg_start[a_start] + ex_arg_end[a_end]
					       + arg_b_lstm);
		}
		if (ex_role_cache.find(role) == ex_role_cache.end()) {
			ex_role_cache[role]
					= arg_tensor_role * affine_transform({arg_b_role, arg_w_role,
			                                          lookup(cg, lookup_params_.at("embed_role_"), role)});
		}
		float binned_len = Bin(a_end - a_start + 1, false);
		float binned_dist = Bin(abs(a_start - p_start) + 1, a_start - p_start < 0);
		Expression ex_len = arg_w_len * binned_len;
		Expression ex_dist = arg_w_dist * binned_dist;
		Expression ex_h = tanh(affine_transform({arg_b_span, arg_w_span,
		                                         ex_span_cache[a_start * slen + a_end]}) +
		                       ex_len + ex_dist);
		Expression ex_span = arg_tensor_span * ex_h;
		ex_scores[r] = pick(arg_out_b, role) +
		               sum_rows(cmult(ex_preds[pred_r],
		                              cmult(ex_span, ex_role_cache[role])));
		(*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
	}
	vector<Expression> i_errs;
	if (!is_train) {
		decoder_->Decode(instance,
		                 parts, *scores, predicted_outputs);
		for (int i = 0; i < parts->size(); ++i) {
			if (!NEARLY_EQ_TOL((*gold_outputs)[i], (*predicted_outputs)[i], 1e-6)) {
				Expression i_err =
						((*predicted_outputs)[i] - (*gold_outputs)[i]) *
						ex_scores[i];
				i_errs.push_back(i_err);
			}
		}
		Expression loss = input(cg, 0.0);
		if (i_errs.size() > 0) {
			loss = loss + sum(i_errs);
		}
		return loss;
	}

	double s_loss = 0.0, s_cost = 0.0;
	int time;
	timeval start, end;
	gettimeofday(&start, NULL);
	// fear
	decoder_->DecodeCostAugmented(instance,
	                              parts, *scores, *gold_outputs,
	                              predicted_outputs,
	                              &s_cost, &s_loss);
	gettimeofday(&end, NULL);
	time = diff_ms(end, start);
	for (int i = 0; i < parts->size(); ++i) {
		if (!NEARLY_EQ_TOL((*gold_outputs)[i], (*predicted_outputs)[i], 1e-6)) {
			Expression i_err = ((*predicted_outputs)[i] - (*gold_outputs)[i]) *
			                   ex_scores[i];
			i_errs.push_back(i_err);
		}
	}
	Expression loss = input(cg, s_cost);
	if (i_errs.size() > 0) {
		loss = loss + sum(i_errs);
	}
	return loss;
}
