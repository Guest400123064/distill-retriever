# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-14-2023
# =============================================================================
"""
This scripts evaluate the specified model on all datasets (hardcoded) in 
    the scripts, or on the specified dataset from command line inputs.
"""

from typing import Dict, Tuple, List, Union

import os
import pathlib

import csv
import json

import argparse

import tqdm
import logging

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

DIR_HOME = pathlib.Path(__file__).parent.parent.absolute()
DIR_EVAL = DIR_HOME / "data" / "evaluations"
DIR_DATA = DIR_HOME / "data" / "datasets"
DATASETS = {"trec-covid",
            "nfcorpus",
            "nq",
            "hotpotqa",
            "fiqa",
            "scidocs",
            "arguana",
            "quora",
            "scifact"}


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
        default="msmarco-MiniLM-L6-cos-v5",
        help="Name of the pre-trained model to use for dense retrieval (default: all-MiniLM-L6-v2).",
    )

    parser.add_argument(
        "--query-encoder",
        type=str,
        default=None,
        help="Name of the pre-trained model to use for query encoding (default: None).",
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
        default="eval_results.csv",
        help="Output file name for evaluation results (default: eval_results.csv).",
    )

    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="Append to the output file if it already exists (default: False).",
    )

    args = parser.parse_args()
    return args


def create_meta_data(document_encoder: str,
                     query_encoder: str, 
                     **kwargs) -> Dict:
    
    from datetime import datetime

    meta = {"datetime":         datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"),
            "document_encoder": document_encoder,
            "query_encoder":    query_encoder}
    
    meta.update(kwargs)
    return meta


def evaluate_on_dataset(dataset_name: str, 
                        document_encoder: str, 
                        query_encoder: str, 
                        batch_size: int) -> Dict[str, Union[str, float]]:
    """Evaluate the specified model on the specified dataset."""

    logging.info(f"Evaluating {query_encoder}:{document_encoder} on {dataset_name}...")

    # Load dataset
    dataset = GenericDataLoader(os.path.join(DIR_DATA, dataset_name))
    corpus, queries, qrels = dataset.load(split="test")

    # Load model
    model_path = document_encoder if query_encoder is None else \
                        (query_encoder, document_encoder)
    model = DRES(models.SentenceBERT(model_path), 
                 batch_size=batch_size)

    # Retrieve
    retriever = EvaluateRetrieval(model, 
                                  score_function="cos_sim", 
                                  k_values=[10, 100])
    results   = retriever.retrieve(corpus, queries)

    # Combine to a single dictionary
    eval_results = create_meta_data(document_encoder, query_encoder, 
                                    dataset_name=dataset_name, 
                                    batch_size=batch_size)
    for metric in retriever.evaluate(qrels, results, 
                                     k_values=retriever.k_values):
        eval_results.update(metric)
    return eval_results


if __name__ == "__main__":
    
    opt = parse_arguments()
    if opt.dataset is None:
        datasets = DATASETS
    else:
        datasets = [opt.dataset]
    
    # Evaluate on all datasets
    eval_results = []
    for dataset in datasets:
        result = evaluate_on_dataset(dataset,
                                     opt.document_encoder,
                                     opt.query_encoder,
                                     opt.batch_size)
        eval_results.append(result)

    # Save results
    target_path = os.path.join(DIR_EVAL, opt.output)
    if opt.append and os.path.exists(target_path):
        logging.info(f"Appending results to {target_path}.")
        f = open(target_path, "a")
    else:
        logging.info(f"Saving (overwrite) results to {opt.output}.")
        f = open(target_path, "w")
    
    writer = csv.DictWriter(f, fieldnames=eval_results[0].keys())
    
    # Write header if file is empty
    if os.stat(target_path).st_size == 0:
        writer.writeheader()
    writer.writerows(eval_results)

    f.close()
