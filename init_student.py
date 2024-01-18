#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initialize student model with various strategies.

This script is used to initialize student model with various strategies:
  - Taking a subset of teacher layers
  - Loading an arbitrary model and align the embedding dimension

The default init strategy is taking a subset of teacher layers. Our empirical
results shows that this strategy surpasses alignment strategy by a large margin.
We fix the pooling strategy to mean pooling."""

from typing import List

import os
from pathlib import Path

import logging

import argparse

import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.models import Transformer


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)


def get_encoder_layers(model: SentenceTransformer) -> nn.Module:
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
    encoder: nn.Module = None

    exc = NotImplementedError(
        f"Not implemented for {auto_model.config.model_type}; ",
        "could not find either `encoder` or `transformer` property",
    )

    if hasattr(auto_model, "encoder"):
        encoder = auto_model.encoder.layer
    elif hasattr(auto_model, "transformer"):
        encoder = auto_model.transformer.layer
    else:
        raise exc

    try:
        return encoder.layers
    except AttributeError:
        raise exc


def take_subset_of_teacher(
    teacher: SentenceTransformer,
    layers: List[int],
) -> SentenceTransformer:
    """"""




def get_args() -> argparse.Namespace:
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        prog="init_student.py",
        description="Initialize student query encoder",
        epilog="python init_student.py",
    )

    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HuggingFace, model card of path to the teacher model",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="subset",
        choices=["subset", "align"],
        help="Student initialization strategy",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output model directory",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Layers to take from the teacher model, starting from 0 ",
            "If `strategy` is `align`, this argument is ignored",
        )
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    pass
