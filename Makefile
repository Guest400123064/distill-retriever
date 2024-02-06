.PHONY: example dist bench speed

example: speed
	@echo "Example pipeline: dist -> bench -> speed"

dist:
	python3 distill.py \
		--teacher msmarco-bert-base-dot-v5 \
		--init-with subset \
		--layers 0 11 \
		--output models/subset-2l \
		--train-batch-size 128 \
		--eval-batch-size 32 \
		--max-seq-length 256 \
		--warmup-steps 1000 \
		--eval-steps 2000 \
		--num-epochs 1 \
		--adamw-lr 1e-4 \
		--adamw-eps 1e-6 \
		--mixed-precision

bench: dist
    python3 benchmark.py \
		--datasets scifact fiqa nfcorpus \
		--document-encoder msmarco-bert-base-dot-v5 \
		--query-encoder models/subset-2l \
		--split test \
		--output evaluations/beir_eval_subset_2l.csv \
		--k-values 10 100 \
		--score-function dot \
		--batch-size 32

speed: bench
	python3 ispeed.py \
		--datasets scifact fiqa nfcorpus \
		--encoders msmarco-bert-base-dot-v5 models/subset-2l \
		--batch-sizes 16 32 64 \
		--max-sentences 1000000 \
		--num-runs 3 \
		--output evaluations/speed_eval.csv
