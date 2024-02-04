#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ispeed.py
# Author: Yuxuan Wang
# Date: 2024-02-04
"""Evaluate the inference speed over BEIR queries."""

from typing import List

import pathlib

import time
import logging
import argparse

import csv
import json

import torch
from sentence_transformers import SentenceTransformer, LoggingHandler


DIR_THIS = pathlib.Path(__file__).resolve().parent
DIR_EVAL = DIR_THIS / "evaluations"
DIR_DATA = DIR_THIS / "benchmarks" / "raw"

DATASETS = [
    "msmarco",                            # General IR (in-domain)  
    "trec-covid", "nfcorpus",             # Bio-medical IR
    "nq", "hotpotqa", "fiqa",             # Question answering
    "scidocs",                            # Citation prediction
    "arguana", "webis-touche2020",        # Argument retrieval
    "quora", "cqadupstack",               # Duplicate question retrieval
    "scifact", "fever", "climate-fever",  # Fact checking
    "dbpedia-entity",                     # Entity retrieval 
]


torch.cuda.empty_cache()
torch.set_num_threads(4)

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def load_queries(dataset: str, max_sentences: int) -> List[str]:

    logging.info(f"Loading queries from {dataset} dataset.")
    path_queries = DIR_DATA / dataset / "queries.jsonl"
    with open(path_queries, "r") as fIn:
        queries = [json.loads(line)["text"] for line in fIn]
    return queries[:max_sentences]


def get_args() -> argparse.Namespace:
    """Config inference speedup evaluations."""

    parser = argparse.ArgumentParser(
        prog="ispeed.py",
        description="Evaluate the inference speed of a sentence encoder.",
        epilog="Email wangy49@seas.upenn.edu for questions."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=False,
        default=None,
        choices=DATASETS,
        help=(
            "Dataset names to evaluate on. All datasets will be used "
            "if not specified (default: None)",
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="msmarco-bert-base-dot-v5",
        help="Name of the sentence encoder model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use",
    )
    parser.add_argument(
        "--max-sentences", 
        type=int,
        default=1_000_000,
        help="Maximum number of sentences to use",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to aggregate over",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_speed.csv",
        help="Path to the output file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    model = SentenceTransformer(args.encoder)
    sentences = load_queries(args.dataset, args.max_sentences)

    logging.info(f"Model Name: {args.encoder}")
    logging.info(f"Number of sentences: {len(sentences)}")
    logging.info(f"Batch size: {args.batch_size}")

    # Discard first run
    start_time = time.time()
    model.encode(sentences, batch_size=args.batch_size, show_progress_bar=False)
    end_time = time.time()

    records = []
    for i in range(args.num_runs):
        logging.info("Run {}".format(i + 1))
        start_time = time.time()
        model.encode(sentences, batch_size=args.batch_size, show_progress_bar=True)
        diff_time = time.time() - start_time
        
        logging.info("Done after {:.2f} seconds".format(diff_time))
        logging.info("Speed: {:.2f} sentences / second".format(len(sentences) / diff_time))
        records.append({
            "encoder": args.encoder,
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "run": i,
            "time": diff_time,
            "speed": len(sentences) / diff_time
        })

    path_output = DIR_EVAL / args.output
    if not path_output.exists():
        with open(DIR_EVAL / args.output, "w") as fOut:
            writer = csv.DictWriter(fOut, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(DIR_EVAL / args.output, "a") as fOut:
            writer = csv.DictWriter(fOut, fieldnames=records[0].keys())
            writer.writerows(records)
