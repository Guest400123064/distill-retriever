#! /bin/bash

# This script evaluates all the models from a list
all_models=(
    'msmarco-bert-base-dot-v5'
    'hlyu/basemodel_1layer_0'
    'hlyu/basemodel_2layer_0_10'
    'hlyu/basemodel_4layer_0_1_10_11'
    'hlyu/msmarco-distilbert-dot-v5'
)

for model in ${all_models[@]}; do
    python scripts/evaluate_encoder.py \
        --dataset nq \
        --encoder $model \
        --batch-size 32
done
