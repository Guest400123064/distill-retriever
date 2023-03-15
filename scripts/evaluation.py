# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-14-2023
# =============================================================================
"""
This scripts evaluate the specified model on all datasets (hardcoded) in 
    the scripts, or on the specified dataset from command line inputs.
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
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES