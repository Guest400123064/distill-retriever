#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: benchmark.py
# Author: Yuxuan Wang
# Date: 2024-02-04
"""Evaluate retrieval performance over the BEIR benchmark."""

import os
import pathlib

import logging
import warnings
import argparse

from collections import deque

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import FlatIPFaissSearch


DIR_THIS = pathlib.Path(__file__).resolve().parent
DIR_EVAL = DIR_THIS / "evaluations"
DIR_BEIR_RAW = DIR_THIS / "benchmarks" / "raw"
DIR_BEIR_IDX = DIR_THIS / "benchmarks" / "faiss"

URL_BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

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


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def download_dataset_from_beir(name: str) -> bool:
    """Download a dataset maintained by the UKP Lab."""

    if os.path.isdir(DIR_BEIR_RAW / name):
        logging.info(f"Dataset {name} already downloaded. Skipping...")
        return True

    try:
        logging.info(f"Downloading dataset {name} from BEIR to {DIR_BEIR_RAW}...")
        util.download_and_unzip(URL_BEIR.format(name=name), DIR_BEIR_RAW)
    except Exception as exc:
        warnings.warn(f"Failed to download dataset {name} from BEIR: {exc}")
        return False

    logging.info(f"Dataset {name} downloaded successfully.")
    return True


def create_faiss_index(args: argparse.Namespace) -> None:
    """Create Faiss indices for a dense retrieval system using the BEIR package."""

    model = SentenceBERT(args.document_encoder)
    index = FlatIPFaissSearch(model, batch_size=args.batch_size)

    queue = deque()  # [ local_save_dir_name... ]
    for name in args.datasets or DATASETS:
        download_dataset_from_beir(name)

        if name != "cqadupstack":
            queue.append(name)
            continue

        # Special handling for the CQADupStack dataset because the dataset has
        # subdirectories for each topic; so we need to flatten the directory.
        for sub_name in os.listdir(DIR_BEIR_RAW / "cqadupstack"):
            sub_name = str(os.path.join("cqadupstack", sub_name))
            queue.append(sub_name)

    while queue:
        ds_name = queue.popleft()
        dir_raw = DIR_BEIR_RAW / ds_name
        dir_idx = DIR_BEIR_IDX / ds_name

        index_file_name = f"{args.document_encoder}.{args.extension}.faiss"
        if os.path.isfile(dir_idx / index_file_name):
            logging.info(f"Index for {ds_name} already exists. Skipping...")
            continue

        corpus, _, _ = GenericDataLoader(dir_raw).load()
        index.index(corpus, score_function=args.score_function)

        dir_idx.mkdir(parents=True, exist_ok=True)
        index.save(dir_idx, prefix=args.document_encoder, ext=args.extension)

        logging.info(f"Index at {dir_idx / index_file_name} created successfully.")
        return None


def get_args() -> argparse.Namespace:
    """Get command line arguments for evaluating distilled query encoder
    performance to compute performance reservation rate."""

    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Evaluate retrieval performance over the BEIR benchmark.",
        epilog="Email wangy49@seas.upenn.edu for questions.",
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
        "--document-encoder",
        type=str,
        required=False,
        default="msmarco-bert-base-dot-v5",
        help="Document encoder model name (default: 'msmarco-bert-base-dot-v5')",
    )
    parser.add_argument(
        "--extension",
        type=str,
        required=False,
        default="flat",
        help="FAISS index extension (default: 'flat')",
    )
    parser.add_argument(
        "--score-function",
        type=str,
        required=False,
        default="dot",
        choices=["cos_sim", "dot"],
        help="Score function to use for similarity measurement (default: 'dot')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16,
        help="Batch size for index creation (default: 16)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    create_faiss_index(args)
