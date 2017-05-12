curr_dir="$PWD"

train_epochs=100
word_dim=25
pre_word_dim=100
pos_dim=25
mlp_dim=100
lstm_dim=200
num_lstm_layers=2
use_pretrained_embedding=true

trainers="adam"
trainer="adam"
formalisms="dm pas psd"
formalism="dm"
language="english"

use_word_dropout=true
word_dropout_rate=0.25

srl_train_cost_false_positives=0.4
srl_train_cost_false_negatives=0.6

parser_file=${curr_dir}/build/neurboparser
file_pretrained_embedding=${curr_dir}/../embedding/glove.100.pruned

file_train=${curr_dir}/../semeval2015_data/${formalism}/data/english/english_${formalism}_augmented_train.sdp
file_dev=${curr_dir}/../semeval2015_data/${formalism}/data/english/english_${formalism}_augmented_dev.sdp
file_test=${curr_dir}/../semeval2015_data/${formalism}/data/english/english_id_${formalism}_augmented_test.sdp
file_model=${curr_dir}/model/${formalism}.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${word_dropout_rate}.model 
file_pruner_model=${curr_dir}/model/${language}_${formalism}.pruner.model
file_prediction=${curr_dir}/prediction/${formalism}.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${word_dropout_rate}.pred
log_file=${curr_dir}/log_test/${formalism}.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${word_dropout_rate}.log

nohup \
${parser_file} --train  --evaluate \
--dynet_mem 512 \
--dynet_seed 823632965 \
--dynet_weight_decay 1e-6 \
--train_epochs=${train_epochs}  \
--file_train=${file_train} \
--file_test=${file_dev}  \
--srl_labeled=true \
--srl_deterministic_labels=true \
--srl_use_dependency_syntactic_features=false \
--srl_prune_labels_with_senses=false \
--srl_prune_labels=true \
--srl_prune_distances=true \
--srl_prune_basic=true \
--srl_train_pruner=false \
--srl_file_pruner_model=${file_pruner_model} \
--srl_pruner_posterior_threshold=0.0001 \
--srl_pruner_max_arguments=20 \
--srl_pruner_train_epochs=10 \
--srl_pruner_train_algorithm=crf_mira \
--srl_pruner_train_regularization_constant=1e12 \
--form_case_sensitive=false \
--srl_model_type=af \
--srl_allow_self_loops=false \
--srl_allow_root_predicate=true \
--srl_allow_unseen_predicates=false \
--srl_use_predicate_senses=false \
--srl_file_format=sdp \
--use_pretrained_embedding=${use_pretrained_embedding} \
--file_pretrained_embedding=${file_pretrained_embedding} \
--word_dim=${word_dim} \
--pre_word_dim=${pre_word_dim} \
--pos_dim=${pos_dim} \
--lstm_dim=${lstm_dim} \
--mlp_dim=${mlp_dim} \
--num_lstm_layers=${num_lstm_layers} \
--use_word_dropout=${use_word_dropout} \
--word_dropout_rate=${word_dropout_rate} \
--trainer=${trainer} \
--srl_train_cost_false_positives=${srl_train_cost_false_positives} \
--srl_train_cost_false_negatives=${srl_train_cost_false_negatives} \
--logtostderr \
--file_model=${file_model} \
--file_prediction=${file_prediction} \
> ${log_file} &


