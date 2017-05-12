curr_dir="$PWD"

train_epochs=100
word_dim=25
pre_word_dim=100
pos_dim=25
label_dim=100
mlp_dim=100
lstm_dim=200
rank=100
num_lstm_layers=2
use_pretrained_embedding=true

output_terms="shared1 shared3 freda1 freda3"
output_term="shared1"
trainer="adam"
use_word_dropout=true
word_dropout_rate=0.25
srl_train_cost_false_positives=0.4
srl_train_cost_false_negatives=0.6
parser_file=${curr_dir}/build/meurboparser
file_pretrained_embedding=${curr_dir}/../embedding/glove.100.pruned
language="english"

task1_file_train=${curr_dir}/../semeval2015_data/dm/data/${language}/${language}_dm_augmented_train.sdp
task1_file_dev=${curr_dir}/../semeval2015_data/dm/data/${language}/${language}_dm_augmented_dev.sdp
task1_file_test=${curr_dir}/../semeval2015_data/dm/data/${language}/${language}_id_dm_augmented_test.sdp
task1_file_pruner_model=${curr_dir}/model/${language}_dm.pruner.model
task1_file_prediction=${curr_dir}/prediction/dm.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.r${rank}.drop${word_dropout_rate}.pred

task2_file_train=${curr_dir}/../semeval2015_data/pas/data/${language}/${language}_pas_augmented_train.sdp
task2_file_dev=${curr_dir}/../semeval2015_data/pas/data/${language}/${language}_pas_augmented_dev.sdp
task2_file_test=${curr_dir}/../semeval2015_data/pas/data/${language}/${language}_id_pas_augmented_test.sdp
task2_file_pruner_model=${curr_dir}/model/${language}_pas.pruner.model
task2_file_prediction=${curr_dir}/prediction/pas.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.r${rank}.drop${word_dropout_rate}.pred

task3_file_train=${curr_dir}/../semeval2015_data/psd/data/${language}/${language}_psd_augmented_train.sdp
task3_file_dev=${curr_dir}/../semeval2015_data/psd/data/${language}/${language}_psd_augmented_dev.sdp
task3_file_test=${curr_dir}/../semeval2015_data/psd/data/${language}/${language}_id_psd_augmented_test.sdp
task3_file_pruner_model=${curr_dir}/model/${language}_psd.pruner.model
task3_file_prediction=${curr_dir}/prediction/psd.${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.r${rank}.drop${word_dropout_rate}.pred

file_model=${curr_dir}/model/${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.r${rank}.drop${word_dropout_rate}.model 
log_file=${curr_dir}/log_test/${trainer}.${output_term}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.r${rank}.drop${word_dropout_rate}.log

nohup \
${parser_file} \
--dynet_mem 512 \
--dynet_seed 823632965 \
--dynet_weight_decay 1e-6 \
--test  --evaluate \
--train_epochs=${train_epochs}  \
--task1_file_train=${task1_file_train} \
--task1_file_test=${task1_file_dev}  \
--srl_task1_file_pruner_model=${task1_file_pruner_model} \
--task1_file_prediction=${task1_file_prediction} \
--task2_file_train=${task2_file_train} \
--task2_file_test=${task2_file_dev}  \
--srl_task2_file_pruner_model=${task2_file_pruner_model} \
--task2_file_prediction=${task2_file_prediction} \
--task3_file_train=${task3_file_train} \
--task3_file_test=${task3_file_dev}  \
--srl_task3_file_pruner_model=${task3_file_pruner_model} \
--task3_file_prediction=${task3_file_prediction} \
--srl_labeled=true \
--srl_deterministic_labels=true \
--srl_use_dependency_syntactic_features=false \
--srl_prune_labels_with_senses=false \
--srl_prune_labels=true \
--srl_prune_distances=true \
--srl_prune_basic=true \
--srl_train_pruner=false \
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
--rank=${rank} \
--num_lstm_layers=${num_lstm_layers} \
--use_word_dropout=${use_word_dropout} \
--word_dropout_rate=${word_dropout_rate} \
--trainer=${trainer} \
--output_term=${output_term} \
--srl_train_cost_false_positives=${srl_train_cost_false_positives} \
--srl_train_cost_false_negatives=${srl_train_cost_false_negatives} \
--logtostderr \
--file_model=${file_model} \
> ${log_file} &

