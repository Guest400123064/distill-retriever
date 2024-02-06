#! /bin/bash

# This script evaluates all the models from a list
all_models=(
    'msmarco-bert-base-dot-v5'
    'hlyu/basemodel_1layer_0'
    'hlyu/basemodel_2layer_0_10'
    'hlyu/basemodel_4layer_0_1_10_11'
)

for model in ${all_models[@]}; do
    for batch_size in 4 8 16 32 64; do
        python scripts/inference_speed.py \
            --dataset nq \
            --encoder $model \
            --batch-size $batch_size
    done
done
