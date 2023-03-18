# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-18-2023
# =============================================================================
"""
This script create FAISS indices datasets with a specified model. It helps 
    to reduce the overhead of re-creating document embeddings every time 
    when evaluating different query encoders.
"""

import os
import pathlib

import argparse

import tqdm
import logging

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalFaissSearch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

DIR_HOME = pathlib.Path(__file__).parent.parent.absolute()
DIR_DATA = DIR_HOME / "data" / "datasets"
DATASETS = ["trec-covid",
            "nfcorpus",
            # "nq",
            # "hotpotqa",
            "fiqa",
            "scidocs",
            "arguana",
            "quora",
            "scifact"]


def parse_arguments():
    """Get command line arguments for index creation script. Mainly specify the 
        dataset name and the model name to create indices with."""

    parser = argparse.ArgumentParser(
        description="Create FAISS indices for a dense retrieval system using the BEIR package."
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        choices=DATASETS,
        help="Dataset name to evaluate on. All datasets will be used if not specified (default: None).",
    )

    parser.add_argument(
        "--document-encoder", "-e",
        type=str,
        default="msmarco-bert-base-dot-v5",
        help="Name of the pre-trained model to use for dense retrieval (default: all-MiniLM-L6-v2).",
    )

    args = parser.parse_args()
    return args


def index_dataset(model, dataset, args):
    """Create FAISS indices for a dataset."""

    logging.info(f"Creating FAISS indices for {dataset} dataset.")
    DIR_INDEX = DIR_HOME / "data" / "indices" / args.document_encoder / dataset
    DIR_INDEX.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_loader = GenericDataLoader(dataset)
    corpus, queries, qrels = data_loader.load(split="test")

    # Create FAISS indices
    faiss_search = DenseRetrievalFaissSearch(model, DIR_INDEX)
    faiss_search.create_index(corpus)