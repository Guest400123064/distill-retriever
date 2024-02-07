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

## __Usage__
### __Scripts__
One can refer to the [`Makefile`](./Makefile) for a simple example workflow that distill and evaluate a two-layer query encoder from [`msmarco-bert-base-dot-v5`](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5). To run the example, type:
```
make example
```

There are three scripts 

### __Data__
We use the [MS MARCO](https://microsoft.github.io/msmarco/) dataset for model distillation. Specifically, we only used the queries from the training set (and dev set for validation). The `distill.py` script will automatically download the dataset and save it under the `downloads/` folder. There should be a copy of the dataset within this repository as well.

### __Distillation__
To train a student model, run the following command:
```
python3 distillation.py
```
The script will train a student model with the default parameters. The arguments are as follows:
```
--teacher_model: the path or huggingface model name of the teacher model, default: 'sentence-transformers/msmarco-bert-base-dot-v5'
--student_model_init: how to initialize student models, default: 'layer_reduction'
--student_model_init_list: the list of huggingface names of the models that student models intialize from, required if student_model_init is not 'layer_reduction'
--output_path: the path to save the student model, default: 'output'
--train_batch_size: the batch size for training, default: 128
--eval_batch_size: the batch size for evaluation, default: 128
```
### __Retrieval Performance Evaluation__
### __Inference Speedup Evaluation__

## __Citing & Authors__
If you find this work helpful, feel free to cite [our publication](https://aclanthology.org/2023.sustainlp-1.23/) submitted to the SustaiNLP workshop at ACL 2023.
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
