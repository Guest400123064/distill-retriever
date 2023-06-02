# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-14-2023
# =============================================================================
"""
This scripts evaluate the specified model on all datasets (hardcoded) in 
    the scripts, or on the specified dataset from command line inputs.
"""

from typing import List, Dict, Union, NoReturn

import os
import pathlib

import csv

import argparse

import tqdm
import logging

from beir import LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

DIR_HOME = pathlib.Path(__file__).parent.parent.absolute()
DIR_EVAL = DIR_HOME / "data" / "evaluations"
DIR_DATA = DIR_HOME / "data" / "datasets"
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
    """Get command line arguments for evaluation script. Mainly specify the 
        dataset name and the model name to evaluate on."""

    parser = argparse.ArgumentParser(
        description="Evaluate a dense retrieval system using the BEIR package."
    )

    parser.add_argument(
        "--dataset", 
        type=str,
        default=None,
        choices=DATASETS,
        help="Dataset name to evaluate on. All datasets will be used if not specified (default: None).",
    )

    parser.add_argument(
        "--document-encoder", 
        type=str,
        default="msmarco-bert-base-dot-v5",
        help="Name of the pre-trained model to use for dense retrieval (default: msmarco-bert-base-dot-v5).",
    )

    parser.add_argument(
        "--query-encoder",
        type=str,
        default=None,
        help="Name of the pre-trained model to use for query encoding (default: None).",
    )

    parser.add_argument(
        "--score-function",
        type=str,
        default="dot",
        choices=["cos_sim", "dot"],
        help="Score function to use for ranking (default: cos_sim).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dense retrieval evaluation (default: 64).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eval_results_faiss.csv",
        help="Output file name for evaluation results (default: eval_results_faiss.csv).",
    )

    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="Append to the output file if it already exists (default: False).",
    )

    return parser.parse_args()


def create_meta_data(document_encoder: str,
                     query_encoder: str, 
                     **kwargs) -> Dict[str, Union[str, float]]:
    
    from datetime import datetime

    meta = {"datetime":         datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"),
            "document_encoder": document_encoder,
            "query_encoder":    query_encoder}
    
    meta.update(kwargs)
    return meta


def evaluate(file, args) -> NoReturn:
    """Evaluate the specified model on the specified dataset."""

    datasets         = args.dataset
    query_encoder    = args.query_encoder
    document_encoder = args.document_encoder
    score_function   = args.score_function
    batch_size       = args.batch_size

    model_path   = document_encoder if query_encoder is None else (query_encoder, document_encoder)
    faiss_search = FlatIPFaissSearch(models.SentenceBERT(model_path), batch_size=batch_size)

    k_values = [10, 100]

    def evaluate_cqadupstack() -> Dict[str, Union[str, float]]:
        """This is a special handler for the CQADupStack dataset. It 
            is composed of multiple sub-datasets, and we evaluate on
            each of them separately and the performances will be averaged."""
        
        logging.info(f"Evaluating {query_encoder}:{document_encoder} on cqadupstack...")

        performances = []
        dir_raw_parent = DIR_DATA / "raw" / "cqadupstack"
        dir_idx_parent = DIR_DATA / "faiss" / "cqadupstack"
        
        prefix  = document_encoder
        ext = "flat"

        for sub_name in os.listdir(dir_raw_parent):
            dir_raw = dir_raw_parent / sub_name
            dir_idx = dir_idx_parent / sub_name

            corpus, queries, qrels = GenericDataLoader(dir_raw).load()
            faiss_search.load(dir_idx, prefix, ext)

            retriever = EvaluateRetrieval(faiss_search, score_function=score_function)
            results   = retriever.retrieve(corpus, queries)

            ret = create_meta_data(document_encoder, query_encoder, 
                                   dataset_name=f"cqadupstack",
                                   score_function=score_function, 
                                   batch_size=batch_size)
            for metric in retriever.evaluate(qrels, results, k_values=k_values):
                ret.update(metric)
            performances.append(ret)

        # Average the performances
        avg = performances[0].copy()
        for key in performances[0].keys():
            data = [p[key] for p in performances]
            if isinstance(data[0], float):
                avg[key] = sum(data) / len(performances)
        return avg

    def evaluate_single(dataset_name: str) -> Dict[str, Union[str, float]]:
        """Evaluate on a single dataset."""
        
        logging.info(f"Evaluating {query_encoder}:{document_encoder} on {dataset_name}...")

        dir_raw = DIR_DATA / "raw" / dataset_name
        dir_idx = DIR_DATA / "faiss" / dataset_name
        prefix  = document_encoder
        ext     = "flat"

        # Load dataset
        corpus, queries, qrels = GenericDataLoader(dir_raw).load()
        faiss_search.load(dir_idx, prefix, ext)

        # Retrieve
        retriever = EvaluateRetrieval(faiss_search, score_function=score_function)
        results   = retriever.retrieve(corpus, queries)

        # Combine to a single dictionary
        ret = create_meta_data(document_encoder, query_encoder, 
                               dataset_name=dataset_name,
                               score_function=score_function, 
                               batch_size=batch_size)
        for metric in retriever.evaluate(qrels, results, k_values=k_values):
            ret.update(metric)
        return ret

    # Evaluate on all datasets
    datasets = DATASETS if datasets is None else [datasets]
    for i, dataset in tqdm.tqdm(enumerate(datasets), 
                                total=len(datasets),
                                desc="Evaluating on datasets"):
        result = evaluate_single(dataset) if dataset != "cqadupstack" else evaluate_cqadupstack()
        if i == 0:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if file.mode == "w": writer.writeheader()
        writer.writerow(result)
        file.flush()
    return


if __name__ == "__main__":
    
    args = parse_arguments()
    eval_file = os.path.join(DIR_EVAL, args.output)
    if args.append and os.path.exists(eval_file):
        logging.info(f"Appending results to {eval_file}.")
        with open(eval_file, "a") as f:
            evaluate(f, args)
    else:
        logging.info(f"Saving (overwrite) results to {eval_file}.")
        with open(eval_file, "w") as f:
            evaluate(f, args)
