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






"""parse command line arguments"""
parser = argparse.ArgumentParser(description='Distill a teacher model into a student model.')
parser.add_argument('--teacher_model', type=str, default='sentence-transformers/msmarco-bert-base-dot-v5')
parser.add_argument('--student_model_options', type=str, default='layer_reduction')
parser.add_argument('--loss_function', type=str, default='mse')
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=64)
# parser.add_argument
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
def load_student_model(construct_mode=args.student_model_options, layers_to_keep=[1, 4, 7, 10]):
    # if layer reduction
    if construct_mode == 'layer_reduction':
        student_model = load_teacher_model()
        auto_model = student_model._first_module().auto_model
        logging.info("Remove layers from student. Only keep these layers: {}".format(layers_to_keep))
        new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
        auto_model.encoder.layer = new_layers
        auto_model.config.num_hidden_layers = len(layers_to_keep)
    else:
         student_model = load_teacher_model(argparse.student_model_options)
    return student_model

teacher_model = load_teacher_model()
student_model = load_student_model()


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


"""define custom evaluator"""

class CosineSimilarityEvaluator(SentenceEvaluator):
    """
    Computes the cosine similarity between 2 models.
    :param eval_sentences: Sentences are embedded with the teacher model
    :param show_progress_bar: Show progress bar when computing embeddings
    :param batch_size: Batch size to compute sentence embeddings
    :param name: Name of the evaluator
    :param write_csv: Write results to CSV file
    """
    def __init__(self, eval_sentences: List[str], teacher_model = None, show_progress_bar: bool = False, batch_size: int = 32, name: str = '', write_csv: bool = True):
        self.eval_embeddings = teacher_model.encode(eval_sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_numpy=True)
        self.eval_sentences = eval_sentences

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "mse_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]
        self.write_csv = write_csv

    def __call__(self, student_model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        student_embeddings = student_model.encode(self.eval_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)

        similarity_all = torch.diagonal(util.cos_sim(self.eval_embeddings, student_embeddings))
        similarity = similarity_all.mean().item()

        logging.info("similarity evaluation (closer to 1 = better) on "+self.name+" dataset"+out_txt)
        logging.info("Similarity:\t{:4f}".format(similarity))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, similarity])

        return similarity

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













