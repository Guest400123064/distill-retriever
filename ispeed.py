#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ispeed.py
# Author: Yuxuan Wang
# Date: 2024-02-04
"""Evaluate the inference speed over BEIR queries."""

from typing import List, Dict
from collections import deque
from itertools import product

import os
import pathlib

import time
import logging
import argparse

import csv
import json

import torch
from sentence_transformers import SentenceTransformer, LoggingHandler


DIR_THIS = pathlib.Path(__file__).resolve().parent
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


torch.set_num_threads(4)

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def load_beir_queries(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Load the queries from the BEIR benchmark."""

    queue = deque()
    for name in args.datasets:
        if name != "cqadupstack":
            queue.append(DIR_DATA / name)
            continue

        for sub_name in os.listdir(DIR_DATA / name):
            queue.append(DIR_DATA / name / sub_name)

    queries = {}
    while queue:
        path = queue.popleft()
        with open(path / "queries.jsonl", "r") as fin:
            queries[path.name] = [
                json.loads(line)["text"] for idx, line in enumerate(fin)
                if idx < args.max_sentences
            ]

    return queries


def evaluate_inference_speed(args: argparse.Namespace) -> None:
    """Evaluate the inference speed of sentence encoders over all
    specified datasets."""

    def _time_once(model, queries, batch_size) -> float:
        start_time = time.time()
        model.encode(queries, batch_size=batch_size, show_progress_bar=False)
        return time.time() - start_time

    datasets = load_beir_queries(args)
    records = []

    for encoder in args.encoders:
        torch.cuda.empty_cache()
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        model = SentenceTransformer(encoder)

        for (name, queries), batch_size in product(datasets.items(), args.batch_sizes):
            logging.info(f"Evaluating {encoder} over {name} with batch size {batch_size}...")

            # Discard the first run as the model needs to be loaded
            # with some overhead that is not representative of the
            # actual inference speed.
            _time_once(model, queries, batch_size)

            # Actual runs
            for i in range(args.num_runs):
                time_elapsed = _time_once(model, queries, batch_size)

                logging.info(f"Run {i + 1} done after {time_elapsed:.2f} seconds")
                records.append({
                    "device":     device,
                    "encoder":    pathlib.Path(encoder).name,
                    "dataset":    name.replace("/", "-"),
                    "batch_size": batch_size,
                    "run":        i,
                    "time":       time_elapsed,
                    "throughput": len(queries) / time_elapsed,
                })

    with open(args.output, "w") as fout:
        writer = csv.DictWriter(fout, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


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
        "--encoders",
        type=str,
        nargs="+",
        required=True,
        help="Name of the sentence encoder models to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Batch sizes to experiment with",
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
    evaluate_inference_speed(args)
