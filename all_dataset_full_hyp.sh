#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Create logs directory if it doesn't exist
if [ ! -d "./logs/log_20250719102426" ]; then
    mkdir ./logs/log_20250719102426
fi

# Create WPMixer logs directory if it doesn't exist
if [ ! -d "./logs/log_20250719102426/WPMixer" ]; then
    mkdir ./logs/log_20250719102426/WPMixer
fi


if [ ! -d "./logs/log_20250719102426/WPMixer/ETTm1" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/ETTm1
fi
if [ ! -d "./logs/log_20250719102426/WPMixer/ETTh1" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/ETTh1
fi
if [ ! -d "./logs/log_20250719102426/WPMixer/ETTm2" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/ETTm2
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/ETTh2" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/ETTh2
fi
if [ ! -d "./logs/log_20250719102426/WPMixer/electricity" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/electricity
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/Exchange" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/Exchange
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/Solar" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/Solar
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/weather" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/weather
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/Traffic" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/Traffic
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/PEMS03" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/PEMS03
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/PEMS04" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/PEMS04
fi

if [ ! -d "./logs/log_20250719102426/WPMixer/PEMS07" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/PEMS07
fi
if [ ! -d "./logs/log_20250719102426/WPMixer/PEMS08" ]; then
    mkdir ./logs/log_20250719102426/WPMixer/PEMS08
fi











#-----------



# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTh1
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000242438 0.000201437 0.000132929 0.000239762)
batches=(256 256 256 256)
wavelets=(db2 db3 db2 db2)
levels=(2 2 1 1)
tfactors=(5 5 3 5)
dfactors=(8 5 3 3)
epochs=(30 30 30 30)
dropouts=(0.4 0.05 0.0 0.2)
embedding_dropouts=(0.1 0.2 0.4 0.4)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 128)
patiences=(12 12 12 12)











# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"
	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done









# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTh2
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000466278 0.000294929 0.000617476 0.000810205)
batches=(256 256 256 256)
wavelets=(db2 db2 db2 db2)
levels=(2 3 3 3)
tfactors=(5 3 5 5)
dfactors=(5 8 3 5)
epochs=(30 30 30 30)
dropouts=(0.0 0.0 0.1 0.4)
embedding_dropouts=(0.1 0.0 0.1 0.0)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 128 128)
patiences=(12 12 12 12)






#		--model=WPMixer
#		--task_name=long_term_forecast
#		--data=ETTh2
#		--seq_len=96
#		--pred_len=336
#		--d_model=128
#		--tfactor=5
#		--dfactor=3
#		--wavelet=db2
#		--level=5
#		--patch_len=16
#		--stride=8
#		--batch_size=256
#		--learning_rate=0.000617476
#		--lradj=type3
#		--dropout=0.1
#		--embedding_dropout=0.1
#		--patience=12
#		--train_epochs=30
#		--use_amp


#
# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do

  echo "$dataset _ ${pred_lens[$i]}"
	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done













#================= level=3的h2

export CUDA_VISIBLE_DEVICES=0






# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm1
seq_lens=( 96)
pred_lens=( 720)
learning_rates=( 0.002011441)
batches=( 256)
wavelets=( db5)
levels=( 3)
tfactors=( 3)
dfactors=( 8)
epochs=( 80)
dropouts=( 0.4)
embedding_dropouts=( 0.05)
patch_lens=( 48)
strides=( 24)
lradjs=( type3)
d_models=( 128)
patiences=( 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"

	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done




#
#
#
#
#
#
# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm1
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.001277976 0.002415901 0.001594735 0.002011441)
batches=(256 256 256 256)
wavelets=(db2 db3 db5 db5)
levels=(1 1 1 4)
tfactors=(5 3 7 3)
dfactors=(3 7 7 8)
epochs=(80 80 80 80)
dropouts=(0.4 0.4 0.4 0.4)
embedding_dropouts=(0.2 0.05 0.0 0.05)
patch_lens=(48 48 48 48)
strides=(24 24 24 24)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 256 128)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"

	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done






# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm2
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.00076587 0.000275775 0.000234608 0.001039536)
batches=(256 256 256 256)
wavelets=(bior3.1 db2 db2 db2)
levels=(1 1 1 1)
tfactors=(3 3 3 3)
dfactors=(8 7 5 8)
epochs=(80 80 80 80)
dropouts=(0.4 0.2 0.4 0.4)
embedding_dropouts=(0.0 0.1 0.0 0.0)
patch_lens=(48 48 48 48)
strides=(24 24 24 24)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 256)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"

	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done




model_name=WPMixer

# Datasets and prediction lengths
dataset=Electricity
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.00328086 0.000493286 0.002505375 0.001977516)
batches=(32 32 32 32)
wavelets=(sym3 coif5 sym4 db2)
levels=(2 3 1 2)
tfactors=(3 7 5 7)
dfactors=(5 5 7 8)
epochs=(100 100 100 100)
dropouts=(0.1 0.1 0.2 0.1)
embedding_dropouts=(0.0 0.05 0.05 0.0)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(32 32 32 32)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"
	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"

	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
#
#
#
#
# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=Traffic
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.0010385 0.000567053 0.001026715 0.001496217)
batches=(16 16 16 16)
wavelets=(db3 db3 bior3.1 db3)
levels=(1 1 1 1)
tfactors=(3 3 7 7)
dfactors=(5 5 7 3)
epochs=(60 60 50 60)
dropouts=(0.05 0.05 0.0 0.05)
embedding_dropouts=(0.05 0.0 0.1 0.2)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(16 32 32 32)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"
	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"

	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
#
#
#
# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=Weather
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000913333 0.001379042 0.000607991 0.001470479)
batches=(32 64 32 128)
wavelets=(db3 db3 db3 db2)
levels=(2 1 2 1)
tfactors=(3 3 7 7)
dfactors=(7 7 7 5)
epochs=(60 60 60 60)
dropouts=(0.4 0.4 0.4 0.4)
embedding_dropouts=(0.1 0.0 0.4 0.2)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 128 128)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
  echo "$dataset _ ${pred_lens[$i]}"
	log_file="logs/log_20250719102426/${model_name}/${dataset}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
