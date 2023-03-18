# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 03-14-2023
# =============================================================================
"""
This scripts downloads the benchmark datasets from BEIR for evaluating the 
    performance of distilled information retrieval models.
"""

import os
import pathlib

import tqdm
import logging

from beir import util, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


DATASET_NAMES = ["trec-covid",
                 "nfcorpus",
                 "nq",
                 "hotpotqa",
                 "fiqa",
                 "scidocs",
                 "arguana",
                 "quora",
                 "scifact"]


if __name__ == "__main__":

    out_dir = pathlib.Path(__file__).parent.parent.absolute() / "data" / "datasets"
    url_fmt = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    for name in tqdm.tqdm(DATASET_NAMES):
        url = url_fmt.format(name=name)
        util.download(url, out_dir / "zip")
        util.unzip(out_dir / "zip" / f"{name}.zip", out_dir / "raw")
