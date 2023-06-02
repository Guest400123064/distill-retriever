# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 04-23-2023
# =============================================================================
"""
This scripts evaluate the inference speed of a specified model on a 
    specified dataset. 
"""

from typing import List

import os
import pathlib

import argparse

import time

import csv
import json

import logging

import torch
torch.set_num_threads(4)  # Limit torch to 4 threads

from beir import LoggingHandler
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

DIR_HOME = pathlib.Path(__file__).parent.parent.absolute()
DIR_DATA = DIR_HOME / "data" / "datasets" / "raw"
DIR_EVAL = DIR_HOME / "data" / "evaluations"
DATASETS = [
    "trec-covid", "nfcorpus",       # Bio-medical IR
    "nq", "fiqa",                   # Question answering
    "scidocs",                      # Citation prediction
    "arguana", "webis-touche2020",  # Argument retrieval
    "quora",                        # Duplicate question retrieval
    "scifact",                      # Fact checking
    "dbpedia-entity"                # Entity retrieval 
]


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Evaluate the inference speed of a sentence encoder."
    )

    parser.add_argument(
        "--dataset", 
        type=str,
        default=None,
        choices=DATASETS,
        help="Name of the dataset to use."
    )

    parser.add_argument(
        "--encoder", "-e",
        type=str,
        default="msmarco-bert-base-dot-v5",
        help="Name of the sentence encoder model to use."
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size to use."
    )

    parser.add_argument(
        "--max-sentences", 
        type=int,
        default=1000000,
        help="Maximum number of sentences to use."
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to aggregate over."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="inference_speed.csv",
        help="Path to the output file."
    )

    args = parser.parse_args()
    return args


def load_queries(dataset: str, max_sentences: int) -> List[str]:

    logging.info(f"Loading queries from {dataset} dataset.")
    path_queries = DIR_DATA / dataset / "queries.jsonl"
    with open(path_queries, "r") as fIn:
        queries = [json.loads(line)["text"] for line in fIn]
    return queries[:max_sentences]


if __name__ == "__main__":

    args = parse_arguments()

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
