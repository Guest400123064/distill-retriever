# __Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost Dense Retriever Online Efficiency__
This repository contains the code for the [Query Encoder Distillation paper](https://aclanthology.org/2023.sustainlp-1.23/).


## __Abstract__
In this work, we propose a minimalistic unsupervised baseline method for enhancing the online efficiency of dual encoder (DE) based text dense information retrieval (IR) systems through query encoder embedding-alignment distillation, i.e., __simply minimizing the Euclidean distance between the student and teacher query embeddings over query text corpus__. We investigate the importance of student initialization, showing that initialization with subset of teacher layers, in particular __top and bottom few layers__, can __improve inference efficiency over 6 times with only 7.5\% average performance degradation__ measured by [the BEIR benchmark](https://github.com/beir-cellar/beir). Our findings aim to increase neural IR system accessibility and prompt the community to reconsider the trade-offs between method complexity and performance improvements.

## __Requirements__
Please use the following command to install the required packages:
```
pip install -r requirements.txt
```
Alternatively, one can leverage `conda` via:
```
conda env create -f environment.yml 
```

__Also, one need sufficient GPU RAM to create embedding indexing for some of the large evaluation corpus like [HotpotQA](https://hotpotqa.github.io/).__ 

## __Usages__
One can refer to the [`Makefile`](./Makefile) for a simple example workflow that distill and evaluate a two-layer query encoder from [`msmarco-bert-base-dot-v5`](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5). To run the example, type:
```
make example
```

Three scripts drives this work. [`distill.py`](./distill.py) initializes a student based on one of the two strategies in the paper as specified, and minimize the embedding Euclidean distances over the MS MARCO queries. The output __directory__ should be specified by the user, usually `models/<YOUR_MODEL_NAME>/`. [`benchmark.py`](./benchmark.py) help download and index BEIR datasets under `benchmarks/`. Two sub folders `benchmarks/raw/` and `benchmarks/faiss/` will be created to store the raw corpus and FAISS indexing. It will then evaluate the specified query encoder with common performance metrics like `nDCG@k`. [`ispeed.py`](./ispeed.py) help evaluate inference throughput on a single CUDA device. All scripts comes with a help page, type:
```
python <SCRIPT_NAME>.py --help
```

The implementation is simple and should be straightforward to read or adapt. However, please feel free to email `wangy49@seas.upenn.edu` for clarification.

## __Training Data__
We use the [MS MARCO](https://microsoft.github.io/msmarco/) dataset for model distillation. Specifically, we only used the queries from the training set (and dev set for validation). The `distill.py` script will automatically download the dataset and save it under the `downloads/` folder. There should be a copy of the dataset within this repository as well.

## __Citing & Authors__
If you find this work helpful, feel free to cite [our publication](https://aclanthology.org/2023.sustainlp-1.23/) submitted to [the SustaiNLP workshop](https://aclanthology.org/volumes/2023.sustainlp-1/) at ACL 2023.
```
@inproceedings{wang-hong-2023-query,
    title = "Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost Dense Retriever Online Efficiency",
    author = "Wang, Yuxuan  and
      Hong, Lyu",
    editor = "Sadat Moosavi, Nafise  and
      Gurevych, Iryna  and
      Hou, Yufang  and
      Kim, Gyuwan  and
      Kim, Young Jin  and
      Schuster, Tal  and
      Agrawal, Ameeta",
    booktitle = "Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sustainlp-1.23",
    doi = "10.18653/v1/2023.sustainlp-1.23",
    pages = "290--298",
}
```
