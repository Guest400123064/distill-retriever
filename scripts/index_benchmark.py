# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-18-2023
# =============================================================================
"""
This script create Faiss indices datasets with a specified model. It helps 
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
from beir.retrieval.search.dense import FlatIPFaissSearch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

DIR_HOME = pathlib.Path(__file__).parent.parent.absolute()
DIR_DATA = DIR_HOME / "data" / "datasets"
URL_DATA = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
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
    """Get command line arguments for index creation script. Mainly specify the 
        dataset name and the model name to create indices with."""

    parser = argparse.ArgumentParser(
        description="Create Faiss indices for a dense retrieval system using the BEIR package."
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

    parser.add_argument(
        "--score-function", "-s",
        type=str,
        default="dot",
        choices=["cos_sim", "dot"],
        help="Score function to use for ranking (default: cos_sim).",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for dense retrieval evaluation (default: 64).",
    )

    return parser.parse_args()


def index(args):
    """Create Faiss indices for specified datasets."""

    datasets   = args.dataset
    model_name = args.document_encoder
    score_func = args.score_function
    batch_size = args.batch_size

    model = models.SentenceBERT(model_name)
    faiss_search = FlatIPFaissSearch(model, batch_size=batch_size)

    def index_single_dataset(dataset_name: str):

        ext = "flat"
        prefix = model_name
        index_file = f"{prefix}.{ext}.faiss"
        dir_out = DIR_DATA / "faiss" / dataset_name
        dir_raw = DIR_DATA / "raw" # / dataset_name
        url_raw = URL_DATA.format(name=dataset_name)

        if os.path.isfile(dir_out / index_file):
            logging.info(f"Faiss index for {dataset_name} with {model_name} already exists.")
            return

        logging.info(f"Loading raw {dataset_name} dataset.")
        util.download_and_unzip(url_raw, dir_raw)

        # The split parameter does not effect the indexing process as we 
        #    only need the corpus; split choose the correct qrel to load 
        corpus, _, _ = GenericDataLoader(dir_raw / dataset_name).load()

        logging.info(f"Creating Faiss index for {dataset_name} dataset with {model_name}...")
        faiss_search.index(corpus, score_function=score_func)

        logging.info(f"Saving Faiss index for {dataset_name} as {index_file}...")
        dir_out.mkdir(parents=True, exist_ok=True)
        faiss_search.save(dir_out, prefix=model_name, ext=ext)

    datasets = DATASETS if datasets is None else [datasets]
    for dataset in tqdm.tqdm(datasets, 
                             total=len(datasets),
                             desc="Indexing datasets"):
        index_single_dataset(dataset)
    return


if __name__ == "__main__":

    index(parse_arguments())
