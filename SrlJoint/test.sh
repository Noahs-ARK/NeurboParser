curr_dir="$PWD"

train_epochs=100
lemma_dim=25
word_dim=100
pos_dim=25
mlp_dim=100
lstm_dim=200
num_lstm_layers=2

pruner_lstm_dim=32
pruner_mlp_dim=32
pruner_num_lstm_layers=1

cost_fp=0.4
cost_fn=0.6
cost_fns="0.6"
ex_fraction=0.0
ex_fractions="0.25"

use_pretrained_embedding=true

trainers="adadelta"
trainer="adam"
formalisms="fn"
formalism="fn"
language="english"

dropout_rate=0.0
dropout_rates="0.0"
word_dropout_rate=0.25

parser_file=${curr_dir}/build/srl_joint
file_pretrained_embedding=${curr_dir}/../embedding/glove.100.pruned

file_exemplar=${curr_dir}/../framenet_data/conll/exemplar
file_frames=${curr_dir}/../framenet_data/conll/frames
file_train=${curr_dir}/../framenet_data/conll/train
file_dev=${curr_dir}/../framenet_data/conll/dev
file_test=${curr_dir}/../framenet_data/conll/test
file_pruner_model=${curr_dir}/model/${language}_${formalism}.pruner.model

file_model=${curr_dir}/model/${trainer}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.fp${cost_fp}.fn${cost_fn}.ex_fraction${ex_fraction}.model 
file_prediction=${curr_dir}/prediction/${trainer}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.fp${cost_fp}.fn${cost_fn}.ex_fraction${ex_fraction}.pred
log_file=${curr_dir}/log/${trainer}.lstm${lstm_dim}.layer${num_lstm_layers}.h${mlp_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.fp${cost_fp}.fn${cost_fn}.ex_fraction${ex_fraction}.log


${parser_file} --train  --evaluate \
--dynet_mem 512 \
--dynet_seed 823632965 \
--dynet_weight_decay 1e-6 \
--train_epochs=${train_epochs} \
--use_exemplar=false \
--exemplar_fraction=${ex_fraction} \
--file_exemplar=${file_exemplar} \
--file_frames=${file_frames} \
--file_train=${file_train} \
--file_test=${file_dev}  \
--file_model=${file_model} \
--file_pruner_model=${file_pruner_model} \
--file_prediction=${file_prediction} \
--srl_labeled=true \
--srl_deterministic_labels=true \
--form_case_sensitive=false \
--srl_model_type=af \
--srl_prune_basic=true \
--srl_train_pruner=false \
--srl_pruner_posterior_threshold=0.00001 \
--use_pretrained_embedding=${use_pretrained_embedding} \
--file_pretrained_embedding=${file_pretrained_embedding} \
--lemma_dim=${lemma_dim} \
--word_dim=${word_dim} \
--pos_dim=${pos_dim} \
--lstm_dim=${lstm_dim} \
--mlp_dim=${mlp_dim} \
--pruner_lstm_dim=${pruner_lstm_dim} \
--pruner_mlp_dim=${pruner_mlp_dim} \
--num_lstm_layers=${num_lstm_layers} \
--pruner_num_lstm_layers=${pruner_num_lstm_layers} \
--dropout_rate=${dropout_rate} \
--word_dropout_rate=${word_dropout_rate} \
--trainer=${trainer} \
--output_term=${output_term} \
--logtostderr \
--max_dist=20 \
--max_span_length=20 \
--srl_train_cost_false_positives=${cost_fp} \
--srl_train_cost_false_negatives=${cost_fn}

