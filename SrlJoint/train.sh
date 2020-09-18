curr_dir="$PWD"

train_epochs=100
lemma_dim=50
word_dim=100
pos_dim=50
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
ex_fractions="0.0 0.25"

use_pretrained_embedding=true

trainers="sgd"
trainer="sgd"
formalism="fn1.7"
language="english"

dropout_rate=0.0
dropout_rates="0.0"
word_dropout_rate=0.25

eta0=0.5 # 0.1 for sgd/adagrad, 1 for adadelta, 0.001 for adam
eta0s="0.5"
eta_decay=0.01
halve=5
halves="10"
batch_size=1
use_elmo=false
file_elmo=${curr_dir}/../embedding/elmo/framenet/


parser_file=${curr_dir}/build/srl_joint
file_pretrained_embedding=${curr_dir}/../embedding/glove.${word_dim}.pruned

file_exemplar=${curr_dir}/../framenet_data/${formalism}/conll/exemplar
file_frames=${curr_dir}/../framenet_data/${formalism}/conll/frames
file_train=${curr_dir}/../framenet_data/${formalism}/conll/train
file_dev=${curr_dir}/../framenet_data/${formalism}/conll/dev
file_test=${curr_dir}/../framenet_data/${formalism}/conll/test
file_pruner_model=${curr_dir}/model/${language}_${formalism}.pruner.model


# pruner_updates: 
# 1.5: 1608
# 1.6: 1860
# 1.7: 1866
for ex_fraction in $ex_fractions; do
	file_model=${curr_dir}/model/${formalism}.${trainer}.bs${batch_size}.eta0${eta0}.halve${halve}.w${word_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.ex_fraction${ex_fraction}.model 
	file_prediction=${curr_dir}/${formalism}.prediction/${trainer}.bs${batch_size}.eta0${eta0}.halve${halve}.w${word_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.ex_fraction${ex_fraction}.pred
	log_file=${curr_dir}/log/${formalism}.${trainer}.bs${batch_size}.eta0${eta0}.halve${halve}.w${word_dim}.drop${dropout_rate}.wdrop${word_dropout_rate}.ex_fraction${ex_fraction}.log

	#nohup \
	GLOG_logtostderr=1 \
    ${parser_file} --train  --evaluate \
	--train_epochs=${train_epochs} \
	--use_exemplar=true \
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
	--max_dist=20 \
	--max_span_length=20 \
	--srl_train_cost_false_positives=${cost_fp} \
	--srl_train_cost_false_negatives=${cost_fn} \
	--pruner_num_updates=1608 \
	--eta0=${eta0} \
	--eta_decay=${eta_decay} \
	--batch_size=${batch_size} \
	--halve=${halve} \
	--use_elmo=${use_elmo} \
	--file_elmo=${file_elmo} \
	#> ${log_file} &
done
