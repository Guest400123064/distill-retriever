# distill-retriever
Distilling bi-encoders for fast ad-hoc query embedding generation. This repository contains the code for the paper [Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost Dense Retriever Online Efficiency](). 

In this work, we propose a minimal baseline method for enhancing the efficiency of DE-based text IR systems through embedding-alignment distillation. We investigate the importance of student initialization, showing that a "well-prepared" student can improve efficiency over 10 times with only 11\% average performance degradation. Our findings aim to increase neural IR system accessibility and prompt the community to reconsider the trade-offs between method complexity and performance improvements. 

## Requirements
Please use the following command to install the required packages:
```
pip install -r requirements.txt
```

## Usage
### Data
We use the [MS MARCO](https://microsoft.github.io/msmarco/) dataset for model distillation. The distilation.py script will automatically download the dataset and save it in the data/ folder.
### Training
To train a student model, run the following command:
```
python3 distillation.py
```
The script will train a student model with the default parameters. The arguments are as follows:
```
--teacher_model: the path or huggingface model name of the teacher model, default: 'sentence-transformers/msmarco-bert-base-dot-v5'
--student_model_options: fill in student model path or hubbingface model name if using a different initialization, otherwise keep default: 'layer_reduction'
--output_path: the path to save the student model, default: 'output'
--train_batch_size: the batch size for training, default: 128
--eval_batch_size: the batch size for evaluation, default: 128
```
### Evaluation


