#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: distill.py
# Author: Yuxuan Wang, Hong Lv
# Date: 2024-02-04
"""Initialize student model with specified strategy and distill over MS MARCO queries.

This script is used to initialize student model with various strategies:
  - Taking a subset of teacher layers
  - Loading an arbitrary model and align the embedding dimension

The default init strategy is taking a subset of teacher layers. Our empirical
results shows that this strategy surpasses alignment strategy by a large margin.
We fix the pooling strategy to mean pooling."""

from typing import List, Tuple

import os
import pathlib
import tarfile

import logging
import warnings
import argparse

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import PreTrainedModel

from sentence_transformers import SentenceTransformer, LoggingHandler, util
from sentence_transformers.models import Transformer, Dense, Pooling
from sentence_transformers.losses import MSELoss
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.evaluation import MSEEvaluator


DIR_THIS = pathlib.Path(__file__).resolve().parent
DIR_DOWN = DIR_THIS / "downloads"
DIR_MSMARCO_QUERIES = DIR_DOWN / "queries"

NAME_MSMARCO_QUERIES_TAR = "queries.tar.gz"
NAME_MSMARCO_QUERIES_EXT_TRAIN = "queries.train.tsv"
NAME_MSMARCO_QUERIES_EXT_VALID = "queries.eval.tsv"

URL_MSMARCO_QUERIES = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"


random.seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def get_ref_of_encoder_and_auto_model(
    model: SentenceTransformer
) -> Tuple[nn.Module, PreTrainedModel]:
    """Get the encoder layers of a sentence transformer model.

    This function is used to get the teacher encoder layers 
    for student initialization. It deals with inconsistencies
    between different HuggingFace transformers model class
    implementations. Specifically, the pretrained models may
    use different properties to store the encoder layers.

    However, this only cover a limited number of cases. For
    instance, the HuggingFace implementation of GPT2 does not
    have a `encoder` or `transformer` property."""

    if not isinstance(model._first_module(), Transformer):
        raise NotImplementedError(
            "Not implemented for non-transformer models"
        )

    auto_model: PreTrainedModel = model._first_module().auto_model
    if hasattr(auto_model, "encoder"):
        encoder = auto_model.encoder
    elif hasattr(auto_model, "transformer"):
        encoder = auto_model.transformer
    else:
        raise NotImplementedError(
            f"Not implemented for {auto_model.config.model_type}; ",
            "could not find either `encoder` or `transformer` property",
        )

    return encoder, auto_model


def init_from_teacher_layers(
    args: argparse.Namespace
) -> Tuple[SentenceTransformer, SentenceTransformer]:
    """Directly load teacher model as the student model, and then override the
    encoder layers with a subset of the initial layers."""

    if args.layers is None:
        raise ValueError("Layers to keep must be specified")

    try:
        student = SentenceTransformer(args.teacher)
        encoder, auto_model = get_ref_of_encoder_and_auto_model(student)
    except Exception as exc:
        logging.error(f"Failed to load teacher model from {args.teacher}: {exc}")
        raise exc

    layers_to_keep_ids = set(args.layers)
    subset_of_layers = nn.ModuleList([
        layer for i, layer in enumerate(encoder.layer)
        if i in layers_to_keep_ids
    ])

    auto_model.config.num_hidden_layers = len(layers_to_keep_ids)
    encoder.layer = subset_of_layers
    return SentenceTransformer(args.teacher), student


def init_from_another_model(
    args: argparse.Namespace
) -> Tuple[SentenceTransformer, SentenceTransformer]:
    """Initialize student model from an arbitrary model, and then align the
    embedding dimension with the teacher model with a dense layer if the two
    dimensions are different."""

    try:
        teacher = SentenceTransformer(args.teacher)
        student_trf = Transformer(args.init_with)
        student_pool = Pooling(student_trf.get_word_embedding_dimension())
    except Exception as exc:
        logging.error(f"Failed to load teacher/student model from {args.teacher}/{args.init_with}: {exc}")
        raise exc

    student_dim = student_pool.get_sentence_embedding_dimension()
    teacher_dim = teacher.get_sentence_embedding_dimension()
    if teacher_dim != student_dim:
        student_proj = Dense(
            in_features=student_dim,
            out_features=teacher_dim,
            activation_function=nn.ReLU(),
        )
        return teacher, SentenceTransformer(modules=[student_trf, student_pool, student_proj])

    return teacher, SentenceTransformer(modules=[student_trf, student_pool])


def load_msmarco_queries() -> Tuple[List[str], List[str]]:
    """Get only the queries from the MS MARCO dataset. Train
    queries are from 'queries.train.tsv' and validation queries
    are from 'queries.eval.tsv'. This is different from our
    initial setup where we randomly partition the train queries
    into 20%-80% for validation and training."""

    path_ext_train = DIR_MSMARCO_QUERIES / NAME_MSMARCO_QUERIES_EXT_TRAIN
    path_ext_valid = DIR_MSMARCO_QUERIES / NAME_MSMARCO_QUERIES_EXT_VALID
    path_tar = DIR_DOWN / NAME_MSMARCO_QUERIES_TAR

    if not os.path.exists(path_ext_train):
        if not os.path.exists(path_tar):
            logging.info(f"Downloading {URL_MSMARCO_QUERIES} to {path_tar}")
            util.http_get(URL_MSMARCO_QUERIES, str(path_tar))

        logging.info(f"Extracting {path_tar} to {DIR_MSMARCO_QUERIES}")
        with tarfile.open(path_tar, "r:gz") as tar:
            tar.extractall(path=DIR_MSMARCO_QUERIES)

    qs_train = set()
    with open(path_ext_train, "r", encoding="utf-8") as fin:
        for line in fin:
            _, query = line.strip().split("\t")
            qs_train.add(query)

    qs_valid = set()
    with open(path_ext_valid, "r", encoding="utf-8") as fin:
        for line in fin:
            _, query = line.strip().split("\t")
            if query not in qs_train:
                qs_valid.add(query)

    qs_train = list(qs_train)
    qs_valid = list(qs_valid)

    logging.info(f"Loaded {len(qs_train)} training queries and {len(qs_valid)} evaluation queries")
    return qs_train, qs_valid


def distill_over_msmarco_queries(args: argparse.Namespace):
    """Initialize student model and distill over MS MARCO queries."""

    init_fn = init_from_teacher_layers
    if args.init_with != "subset":
        init_fn = init_from_another_model

    teacher, student = init_fn(args)
    qs_train, qs_valid = load_msmarco_queries()

    ds_train = ParallelSentencesDataset(
        student, teacher,
        batch_size=args.train_batch_size,
        use_embedding_cache=(args.num_epochs > 1),
    )
    ds_train.add_dataset(
        [[s] for s in qs_train],
        max_sentence_length=args.max_seq_length,
    )
    dl_train = DataLoader(
        ds_train,
        shuffle=(not args.keep_order),
        batch_size=args.train_batch_size,
    )

    evaluator = MSEEvaluator(
        qs_valid, qs_valid,
        teacher_model=teacher,
        batch_size=args.eval_batch_size,
        show_progress_bar=True,
        name="msmarco-dev-queries",
        write_csv=False,
    )

    loss = MSELoss(model=student)
    student.fit(
        train_objectives=[(dl_train, loss)],
        epochs=args.num_epochs,
        evaluator=evaluator,
        evaluation_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        optimizer_params={
            "lr": args.adamw_lr,
            "eps": args.adamw_eps,
        },
        output_path=args.output,
        save_best_model=True,
        use_amp=args.mixed_precision,
    )


def get_args() -> argparse.Namespace:
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        prog="distill.py",
        description="Distill a large teacher query encoder to a slim student.",
        epilog="Email wangy49@seas.upenn.edu for questions.",
    )

    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HuggingFace model card of path to the teacher model",
    )
    parser.add_argument(
        "--init-with",
        type=str,
        default="subset",
        help=(
            "Student initialization strategy; "
            "default to `subset` (taking a subset of teacher layers), "
            "otherwise, the argument would be considered the HuggingFace "
            "model card or path to the student model if stored locally"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the student model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=False,
        help=(
            "Layers to take from the teacher model, starting from 0; "
            "if `--init-with` is a path or mode card, this argument is ignored"
        ),
    )
    parser.add_argument(
        "--keep-order",
        required=False,
        action="store_true",
        help="Keep the order of the queries in the dataset unchanged",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        required=False,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        required=False,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=False,
        default=256,
        help="Maximum sequence length for the input",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        required=False,
        default=2000,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train the student model",
    )
    parser.add_argument(
        "--adamw-lr",
        type=float,
        required=False,
        default=1e-4,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--adamw-eps",
        type=float,
        required=False,
        default=1e-6,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--mixed-precision",
        required=False,
        action="store_true",
        help="Use mixed precision training 'use_amp=True'",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Suppress UserWarning raised from SentenceTransformers
    # ParallelSentencesDataset 'Creating a tensor from a list of numpy.ndarrays is extremely slow.'
    # This cannot be resolved without modifying the sentence-transformers source code.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        distill_over_msmarco_queries(args)
