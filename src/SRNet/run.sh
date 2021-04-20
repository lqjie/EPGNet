#!/bin/bash

model_name="EPG_SRNet" # model name
save_name="EPG_SRNet_juni04" # checkpoint folder
data_split='data_split_80k' # data split file name (json file)
gpuNum=0
is_jpeg=1 # 0-spatial domain   1-jpeg domain

python train.py \
    --cover_path /media/steg/ALASKA/ALASKA_v2_JPG_256_QF75_GrayScale_dec/ \
    --stego_path /media/steg/alaska_jpeg/juni75/stego_juni04_dec/ \
    --cover_beta_path /media/steg/alaska_jpeg/juni75/cover_juni04_beta/ \
    --stego_beta_path /media/steg/alaska_jpeg/juni75/stego_juni04_beta/ \
    --model_name ${model_name} \
    --data_split ${data_split} \
    --LOG_DIR ${save_name} \
    --batch_size 32 \
    --is_jpeg ${is_jpeg} \
    --max_iter 125000 \
    --step_iter 100000 \
    --valid_interval 2500 \
    --init_lr 0.001 \
    --step_lr 0.0001 \
    --gpu ${gpuNum} \

