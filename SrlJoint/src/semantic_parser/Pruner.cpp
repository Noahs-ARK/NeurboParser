//
// Created by hpeng on 7/9/17.
//

#include "Pruner.h"

void Pruner::InitParams(ParameterCollection *model) {
    // shared
	lookup_params_ = {
			{"embed_word_", model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM})},
			{"embed_lemma_", model->add_lookup_parameters(LEMMA_SIZE, {LEMMA_DIM})},
			{"embed_pos_", model->add_lookup_parameters(POS_SIZE, {POS_DIM})},
			{"embed_pred_length_", model->add_lookup_parameters(5, {4})},
			{"embed_lu_", model->add_lookup_parameters(LU_SIZE, {LU_DIM})},
			{"embed_lu_pos_", model->add_lookup_parameters(LU_SIZE, {POS_DIM})},
			{"embed_frame_", model->add_lookup_parameters(FRAME_SIZE, {FRAME_DIM})},
			{"embed_unlab_length_", model->add_lookup_parameters(5, {4})}
	};

	params_ = {
			{"pred_w_len_", model->add_parameters({MLP_DIM, 1})},
			{"pred_w_lstm_", model->add_parameters({MLP_DIM, 4 * LSTM_DIM})},
			{"pred_b_lstm_", model->add_parameters({MLP_DIM})},
			{"pred_w_span_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"pred_b_span_", model->add_parameters({MLP_DIM})},
			{"pred_tensor_span_", model->add_parameters({RANK, MLP_DIM})},
			{"pred_w_lu_", model->add_parameters({MLP_DIM, LU_DIM + POS_DIM})},
			{"pred_b_lu_", model->add_parameters({MLP_DIM})},
			{"pred_tensor_lu_", model->add_parameters({RANK, MLP_DIM})},
			{"pred_w_frame_", model->add_parameters({MLP_DIM, FRAME_DIM})},
			{"pred_b_frame_", model->add_parameters({MLP_DIM})},
			{"pred_tensor_frame_", model->add_parameters({RANK, MLP_DIM})},
			{"pred_b_out_", model->add_parameters({FRAME_SIZE})},

			{"unlab_w_len_", model->add_parameters({MLP_DIM, 1})},
			{"unlab_w_dist_", model->add_parameters({MLP_DIM, 1})},
			{"unlab_w_start_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_w_end_", model->add_parameters({MLP_DIM, 2 * LSTM_DIM})},
			{"unlab_b_lstm_", model->add_parameters({MLP_DIM})},
			{"unlab_w_span_", model->add_parameters({MLP_DIM, MLP_DIM})},
			{"unlab_b_span_", model->add_parameters({MLP_DIM})},
			{"unlab_tensor_span_", model->add_parameters({RANK, MLP_DIM})},
			{"unlab_b_out_", model->add_parameters({1})},
	};
}

void Pruner::StartGraph(ComputationGraph &cg, bool is_train) {
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
	for (auto it = params_.begin(); it != params_.end(); ++ it) {
		cg_params_[it->first] = parameter(cg, it->second);
	}
}

Expression Pruner::BuildGraph(Instance *instance,
                              Parts *parts, vector<double> *scores,
                              const vector<double> *gold_outputs, vector<double> *predicted_outputs,
                              unordered_map<int, int> *form_count,
                              bool is_train, ComputationGraph &cg) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int offset_pred_parts, num_pred_parts;
    semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
    int offset_unlab_arg_parts, num_unlab_arg_parts;
    semantic_parts->GetOffsetUnlabeledArgument(&offset_unlab_arg_parts, &num_unlab_arg_parts);

    const int slen = sentence->size();
    const vector<int> words = sentence->GetFormIds();
    const vector<int> lemmas = sentence->GetLemmaIds();
    const vector<int> pos = sentence->GetPosIds();
    l2rbuilder_.start_new_sequence();
    r2lbuilder_.start_new_sequence();

	Expression pred_w_lstm = cg_params_.at("pred_w_lstm_");
	Expression pred_b_lstm = cg_params_.at("pred_b_lstm_");
	Expression pred_w_span = cg_params_.at("pred_w_span_");
	Expression pred_b_span = cg_params_.at("pred_b_span_");
	Expression pred_tensor_span = cg_params_.at("pred_tensor_span_");

	Expression pred_w_len = cg_params_.at("pred_w_len_");
	Expression pred_w_lu = cg_params_.at("pred_w_lu_");
	Expression pred_b_lu = cg_params_.at("pred_b_lu_");
	Expression pred_tensor_lu = cg_params_.at("pred_tensor_lu_");
	Expression pred_w_frame = cg_params_.at("pred_w_frame_");
	Expression pred_b_frame = cg_params_.at("pred_b_frame_");
	Expression pred_tensor_frame = cg_params_.at("pred_tensor_frame_");

	Expression unlab_w_len = cg_params_.at("unlab_w_len_");
	Expression unlab_w_dist = cg_params_.at("unlab_w_dist_");
	Expression unlab_w_start = cg_params_.at("unlab_w_start_");
	Expression unlab_w_end = cg_params_.at("unlab_w_end_");
	Expression unlab_b_lstm = cg_params_.at("unlab_b_lstm_");
	Expression unlab_w_span = cg_params_.at("unlab_w_span_");
	Expression unlab_b_span = cg_params_.at("unlab_b_span_");
	Expression unlab_tensor_span = cg_params_.at("unlab_tensor_span_");

	Expression pred_b_out = cg_params_.at("pred_b_out_");
	Expression unlab_b_out = cg_params_.at("unlab_b_out_");

    vector<Expression> ex_words(slen), ex_l2r(slen), ex_r2l(slen), ex_lstm(slen);
	vector<Expression> ex_unlab_start(slen), ex_unlab_end(slen);
    for (int i = 0; i < slen; ++i) {
        int word_idx = words[i];
        int lemma_idx = lemmas[i];
        if (is_train && WORD_DROPOUT > 0.0 && word_idx != UNK_ID) {
            int count = form_count->find(word_idx)->second;
            float rand_float = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
            if (rand_float < WORD_DROPOUT / (static_cast<float> (count) + WORD_DROPOUT)) {
                word_idx = UNK_ID;
                lemma_idx = UNK_ID;
            }
        }
        Expression x_word = lookup(cg, lookup_params_.at("embed_word_"), word_idx);
        Expression x_lemma = lookup(cg, lookup_params_.at("embed_lemma_"), lemma_idx);
        Expression x_pos = lookup(cg, lookup_params_.at("embed_pos_"), pos[i]);
//        ex_words[i] = affine_transform({b_lstm_in, w_lstm_in,
//                                        concatenate({x_word, x_lemma, x_pos})});
	    ex_words[i] = concatenate({x_word, x_lemma, x_pos});
	    if (is_train && DROPOUT > 0) {
		    ex_words[i] = dropout(ex_words[i], DROPOUT);
	    }
        ex_l2r[i] = l2rbuilder_.add_input(ex_words[i]);
    }
    for (int i = 0; i < slen; ++i) {
        ex_r2l[slen - i - 1] = r2lbuilder_.add_input(ex_words[slen - i - 1]);
	    ex_lstm[slen - i - 1] = concatenate({ex_l2r[slen - i - 1], ex_r2l[slen - i - 1]});
	    ex_unlab_start[slen - i - 1] = unlab_w_start *  ex_lstm[slen - i - 1];
	    ex_unlab_end[slen - i - 1] = unlab_w_end *  ex_lstm[slen - i - 1];
    }

    vector<Expression> ex_scores(parts->size());
    scores->assign(parts->size(), 0.0);
    predicted_outputs->assign(parts->size(), 0.0);

    vector<Expression> ex_predicates(num_pred_parts);

	int p_start, p_end, a_start, a_end;
	auto pred_part = static_cast<SemanticPartPredicate *> ((*parts)[offset_pred_parts]);
	pred_part->span(p_start, p_end);

	{
		float binned_len = Bin(p_end - p_start + 1, false);
		Expression ex_len = pred_w_len * binned_len;
		Expression ex_i = tanh(affine_transform({pred_b_lstm, pred_w_lstm,
		                                         concatenate({ex_lstm[p_start], ex_lstm[p_end]})}));
		Expression ex_h = tanh(affine_transform({pred_b_span, pred_w_span, ex_i}) + ex_len);
		Expression ex_pred_span = pred_tensor_span * ex_h;

		for (int p = 0; p < num_pred_parts; ++p) {
			auto pred_part = static_cast<SemanticPartPredicate *> ((*parts)[p + offset_pred_parts]);

			int lu_name_id = pred_part->lu_name();
			int lu_pos_id = pred_part->lu_pos();
			int frame_id = pred_part->frame();

			Expression ex_lu = concatenate({lookup(cg, lookup_params_.at("embed_lu_"), lu_name_id),
			                                lookup(cg, lookup_params_.at("embed_lu_pos_"), lu_pos_id)});
			Expression ex_pred_lu = pred_tensor_lu
			                        * affine_transform({pred_b_lu, pred_w_lu, ex_lu});
			Expression ex_frame = lookup(cg, lookup_params_.at("embed_frame_"), frame_id);
			Expression ex_pred_frame = pred_tensor_frame
			                           * affine_transform({pred_b_frame, pred_w_frame, ex_frame});

			ex_predicates[p] = cmult(ex_pred_span,
			                         cmult(ex_pred_lu, ex_pred_frame));
			ex_scores[p + offset_pred_parts] = pick(pred_b_out, frame_id) +
			                                   sum_rows(ex_predicates[p]);
			(*scores)[p + offset_pred_parts] = as_scalar(cg.incremental_forward(ex_scores[p + offset_pred_parts]));
		}
	}


    for (int i = 0; i < num_unlab_arg_parts; ++i) {
        int r = i + offset_unlab_arg_parts;
        auto arg = static_cast<SemanticPartArgument *>((*parts)[r]);
        int pred_idx = arg->pred_idx();
        arg->span(a_start, a_end);

	    float binned_len = Bin(a_end - a_start + 1, false);
	    float binned_dist = Bin(abs(a_start - p_start) + 1, a_start - p_start < 0);
	    Expression ex_len = unlab_w_len * binned_len;
	    Expression ex_dist = unlab_w_dist * binned_dist;
	    Expression ex_i = tanh(ex_unlab_start[a_start] + ex_unlab_end[a_end]
	                           + unlab_b_lstm);
	    Expression ex_h = tanh(affine_transform({unlab_b_span, unlab_w_span, ex_i})
	                           + ex_len + ex_dist);
        Expression ex_span = unlab_tensor_span * ex_h;
        ex_scores[r] = unlab_b_out +
		        sum_rows(cmult(ex_predicates[pred_idx], ex_span));
        (*scores)[r] = as_scalar(cg.incremental_forward(ex_scores[r]));
    }
    Expression entropy = input(cg, 0.0);
    DecodeBasicMarginals(instance, parts, ex_scores,
                         predicted_outputs, entropy, cg);
    if (is_train) {
        vector<Expression> i_errs;
        for (int r = 0; r < parts->size(); ++r) {
            if ((*gold_outputs)[r] != (*predicted_outputs)[r]) {
                Expression i_err = ((*predicted_outputs)[r] - (*gold_outputs)[r]) * ex_scores[r];
                i_errs.push_back(i_err);
            }
        }
        if (i_errs.size() > 0) {
            entropy = entropy + sum(i_errs);
        }
    }

//    LOG(INFO) <<"Pred: "<< num_pred_parts;
//    for (int i = 0;i < num_pred_parts; ++ i) {
//        int r = i + offset_pred_parts;
//        if ((*gold_outputs)[r] > 0.5 || (*predicted_outputs)[r] > 0.5 || true) {
//            LOG(INFO) << r <<" " << (*scores)[r] <<" "
//                      << (*gold_outputs)[r] <<" "<< (*predicted_outputs)[r];
//
//        }
//    }
//    LOG(INFO) <<"Arg: " << num_unlab_arg_parts;
//    int a_s, a_e;
//    for (int i = 0;i < num_unlab_arg_parts; ++ i) {
//        int r = i + offset_unlab_arg_parts;
//        SemanticPartArgument *arg
//                = static_cast<SemanticPartArgument *> ((*parts)[r]);
//        arg->span(a_s, a_e);
//        if ((*gold_outputs)[r] > 0.5 || (*predicted_outputs)[r] > 0.5) {
//            LOG(INFO) << r <<" " << a_s<<" " << a_e<<" " << arg->pred_idx()<<" "<< (*scores)[r] <<" "
//                      << (*gold_outputs)[r] <<" "<< (*predicted_outputs)[r];
//
//        }
//    }
//    double e = as_scalar(cg.incremental_forward(entropy));
//    LOG(INFO) << "extropy: "<<e<<endl;

//    double l = as_scalar(cg.incremental_forward(entropy));
//    LOG(INFO) << "loss: "<<l<<endl;
//    LOG(INFO) << endl;
    return entropy;
}

void Pruner::DecodeBasicMarginals(Instance *instance, Parts *parts,
                                  const vector<Expression> &scores,
                                  vector<double> *predicted_output,
                                  Expression &entropy, ComputationGraph &cg) {
    SemanticInstanceNumeric *sentence =
            static_cast<SemanticInstanceNumeric *>(instance);
    SemanticParts *semantic_parts = static_cast<SemanticParts *>(parts);
    int offset_pred_parts, num_pred_parts;
    semantic_parts->GetOffsetPredicate(&offset_pred_parts, &num_pred_parts);
    int offset_unlab_arg_parts, num_unlab_arg_parts;
    semantic_parts->GetOffsetUnlabeledArgument(&offset_unlab_arg_parts, &num_unlab_arg_parts);

    predicted_output->assign(parts->size(), 0.0);

    Expression log_partition_function = input(cg, 0.0);
    Expression log_partition_all_frames = input(cg, 0.0);
    vector<Expression> log_partition_frames(num_pred_parts);
    vector<vector<Expression> > log_partition_args(num_pred_parts);

    for (int i = 0;i < num_pred_parts; ++ i) {
        int pred_r = offset_pred_parts + i;
        Expression score = scores[pred_r];

        vector<int> args_by_predicate
                = semantic_parts->GetArgumentsByPredicate(pred_r);
        log_partition_args[i].assign(args_by_predicate.size(),
                                     input(cg, 0.0));
        for (int j = 0;j < args_by_predicate.size(); ++ j) {
            int arg_r = args_by_predicate[j];
            log_partition_args[i][j]
                    = logsumexp({log_partition_args[i][j], scores[arg_r]});
            score = score + log_partition_args[i][j];
        }
        log_partition_frames[i] = score;
        log_partition_all_frames = logsumexp({log_partition_all_frames, log_partition_frames[i]});

    }
    if (num_pred_parts > 0) {
        log_partition_function = log_partition_function + log_partition_all_frames;
    }
    for (int i = 0;i < num_pred_parts; ++ i) {
        int pred_r = offset_pred_parts + i;
        Expression pred_marginal = exp(log_partition_frames[i] - log_partition_all_frames);
        (*predicted_output)[pred_r]
                = as_scalar(cg.incremental_forward(pred_marginal));
        entropy = entropy - scores[pred_r] * pred_marginal;

        vector<int> args_by_predicate
                = semantic_parts->GetArgumentsByPredicate(pred_r);
        for (int j = 0;j < args_by_predicate.size(); ++ j) {
            int arg_r = args_by_predicate[j];
            Expression marginal = exp(scores[arg_r] -  log_partition_args[i][j]);
            marginal = marginal * pred_marginal;
            (*predicted_output)[arg_r]
                    = as_scalar(cg.incremental_forward(marginal));
            entropy = entropy - scores[arg_r] * marginal;
        }
    }
    entropy = entropy + log_partition_function;
}
