## Training and Fine-tuning of Large Language Models
This repository contains examples of training and fine-tuning of LLMs for text classification, image to text, and question answering.

## Requirement
```text
python>=3.8
torch>=2.0.1
numpy>=1.23.5
transformers>=4.37.2
torchvision>=0.17.0
datasets>=2.16.1
evaluate>=0.4.1
huggingface_hub>=0.20.3
jiwer>=3.0.3
```

## Install

Install the required libraries by running: 

```commandline
pip install huggingface-cli
pip install torch torchvision
pip install numpy
pip install transformers datasets evaluate -q
pip install jiwer -q
```
## Login to huggingface
First, you need to create a token by creating an account in huggingface, to be able to access the datasets available on huggingface hub. Then you can use the token login.
```commandline
huggingface-cli login
```

## Command Line Arguments
```text
- fine_tune (bool): whether to perform fine-tuning or just train the base models.
- fine_tune_type (int): the type of fine-tuning on base model. 
      Choices: 1 for multi-scale sequential modeling (ATA) and 2 for improved coarse and fine-grained feature learning. 
- batch_size (int): size of each batch  
```

## How to Run
### Question/Answer Example
#### train base model 
```commandline
python train_question_answering.py --fine_tune False --batch_size 4
```
### train and fine-tune: type 1
```commandline
python train_question_answering.py --fine_tune True --fine_tune_type 1 --batch_size 4
```
