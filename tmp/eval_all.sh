#! /bin/zsh

# This script evaluates all the models from a list
all_models=("msmarco-bert-base-dot-v5"
            "")
doc_encoder="msmarco-bert-base-dot-v5"
evals_file='eval_results_full.csv'

for model in $all_models; do
    echo "Evaluating query encoder < $model >"
    python scripts/evaluate_encoder.py \
        --document-encoder $doc_encoder \
        --query-encoder $model \
        --output $evals_file \
        --append \
        --batch-size 128
done
