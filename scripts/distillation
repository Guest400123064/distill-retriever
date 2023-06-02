import tarfile
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import models, losses, evaluation
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.datasets import ParallelSentencesDataset
import logging
from datetime import datetime
import os
import gzip
import csv
import random
import torch
import argparse
from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List
from torch import nn
import random
import itertools
import time
random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"




"""parse command line arguments"""
parser = argparse.ArgumentParser(description='Distill a teacher model into a student model.')
parser.add_argument('--teacher_model', type=str, default='sentence-transformers/msmarco-bert-base-dot-v5')
parser.add_argument('--student_model_options', type=str, default='layer_reduction')
parser.add_argument('--loss_function', type=str, default='mse')
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=128)
args = parser.parse_args()


"""set up logging"""
# debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


"""load teacher and student models"""
# load teacher model
def load_teacher_model(model_name=args.teacher_model):
    model = SentenceTransformer(model_name)
    return model

# load student model
def load_student_model(construct_mode, teacher_model=None, layers_to_keep=[1, 4, 7, 10]):
    # if layer reduction
    if construct_mode == 'layer_reduction':
        if teacher_model is None:
            student_model = load_teacher_model()
        else:
            student_model = teacher_model
        auto_model = student_model._first_module().auto_model
        # extract layers to keep
        if auto_model.base_model_prefix == "bert" or auto_model.base_model_prefix == "mpnet":
            new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
            auto_model.encoder.layer = new_layers
        elif auto_model.base_model_prefix == "distilbert":
            new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.transformer.layer) if i in layers_to_keep])
            auto_model.transformer.layer = new_layers
        logging.info("Remove layers from student. Only keep these layers: {}".format(layers_to_keep))
        auto_model.config.num_hidden_layers = len(layers_to_keep)
    else:
        student_model_t = models.Transformer(construct_mode, max_seq_length=256)
        student_model_p = models.Pooling(student_model_t.get_word_embedding_dimension())
        # if student model has different hidden size, change it to match teacher model
        if student_model_p.get_sentence_embedding_dimension() != teacher_model.get_sentence_embedding_dimension():
            logging.info("Adding dense layer to student model to match teacher model's embedding size.")
            teacher_embedding_dim = teacher_model.get_sentence_embedding_dimension()
            student_embedding_dim = student_model_p.get_sentence_embedding_dimension()
            # if distilbert, add normalization layer
            if "distilbert" in construct_mode:
                student_model = SentenceTransformer(modules=[student_model_t, student_model_p, 
                                                             models.Normalize(),
                                                             models.Dense(in_features=student_embedding_dim, out_features=teacher_embedding_dim, activation_function=nn.ReLU()), ])
            else:
                student_model = SentenceTransformer(modules=[student_model_t, student_model_p, 
                                                         models.Dense(in_features=student_embedding_dim, out_features=teacher_embedding_dim, activation_function=nn.ReLU()), ])
        else:
            # put components together if same dimension
            student_model = SentenceTransformer(modules=[student_model_t, student_model_p])
    return student_model


teacher_model = load_teacher_model()


torch.cuda.empty_cache()
"""load query data to train student model"""
data_folder = 'msmarco-data'
# Read the train queries into a list
queries = []      
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries.append(query)
queries = list(set(queries))
random.shuffle(queries)
# log number of queries
train_queries = queries[:int(len(queries)*0.8)]
eval_queries = queries[int(len(queries)*0.8):]
logging.info("Train queries: {}".format(len(train_queries)))
logging.info("Eval queries: {}".format(len(eval_queries)))


sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator_sts = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

logging.info("Teacher model performance over STS benchmark : {}".format(dev_evaluator_sts(teacher_model)))
logging.info("Student model performance over STS benchmark : {}".format(dev_evaluator_sts(student_model)))


"""create dataset, data loader, and loss function for training student model"""
# Use parallel sentences dataset to train the student model
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=args.batch_size)
train_data.add_dataset([[sent] for sent in train_queries], max_sentence_length=256)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
if args.loss_function == 'mse':
    train_loss = losses.MSELoss(student_model, teacher_model)
elif args.loss_function == 'cosine':
    train_loss = losses.CosineSimilarityLoss(student_model, teacher_model)
else:
    train_loss = losses.MSELoss(student_model, teacher_model) + losses.CosineSimilarityLoss(student_model, teacher_model)
eval_evaluator_mse = evaluation.MSEEvaluator(eval_queries, eval_queries, teacher_model=teacher_model)


def extract_layers_and_train(teacher_model, student_layers, original_model=None):
    model_name_ = "basemodel" if original_model is None else original_model
    if not os.path.exists(model_name_):
        os.makedirs(model_name_)
    for i, layer_list in enumerate(student_layers):
        # create student model
        student_model = load_student_model("layer_reduction", None, layer_list)
        model_name = model_name_ + "_{}layer".format(len(layer_list))
        for layer in layer_list:
            model_name += "_{}".format(layer)
        print("===============")
        print("+++++++   {}th model:   ".format(i) + model_name + "   ++++++++")
        # test similarity
        print("Similarity score with sts-b embedding before training: {:.4f}".format(dev_evaluator_sts(student_model)))
        print("Euclidean similarity score with teacher embedding before training: {:.4f}".format(eval_query_sim(student_model, output_path=model_name_, epoch=i, steps=0)))
        # prepare data
        train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=128)
        train_data.add_dataset([[sent] for sent in train_queries], max_sentence_length=256)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=128)
        train_loss = losses.MSELoss(model=student_model)
        # train model
        output_path = model_name_+ "/output/" + model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluation.SequentialEvaluator([eval_evaluator_mse, dev_evaluator_sts]),
                      epochs=1,
                      warmup_steps=1000,
                      evaluation_steps=2000,
                      output_path=output_path,
                      save_best_model=True,
                      optimizer_params={'lr': 1e-4, 'eps': 1e-6},
                      use_amp=True)
        # test similarity
        print("Similarity score with sts-b embedding after training: {:.4f}".format(dev_evaluator_sts(student_model)))
        print("Euclidean similarity score with teacher embedding after training: {:.4f}".format(eval_query_sim(student_model, output_path=model_name_, epoch=i, steps=1)))
        student_model.save_to_hub = new_save_to_hub
        student_model.save_to_hub(student_model, repo_name=model_name, exist_ok=True)
        print("===============\n\n")
        
base_model_extraction_scheme = [[1, 4, 7, 10],
                                [0, 1, 10, 11],
                                [0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11],
                                [0, 10],
                                [0, 11],
                                [1, 10],
                                [1, 11],
                                [0], [1], [10], [11]]
                                
                                
extract_layers_and_train(teacher_model, base_model_extraction_scheme)

"""train student model"""
output_path = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=eval_evaluator_mse,
                  epochs=1,
                  warmup_steps=1000,
                  evaluation_steps=5000,
                  output_path=output_path,
                  save_best_model=True,
                  optimizer_params={'lr': 1e-4, 'eps': 1e-6},
                  use_amp=True)

logging.info("Teacher model performance over STS benchmark : {}".format(dev_evaluator_sts(teacher_model)))
logging.info("Student model performance over STS benchmark : {}".format(dev_evaluator_sts(student_model)))













