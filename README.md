# OPN-masterthesis
## Table of Contents

1. Introduction
1. Requirements and Dependencies
1. Models and Training Data
1. Training
1. Testing

## Introduction
This repository is part of my master thesis project _"Temporal order prediction of video frames in intelligent machine applications using self-supervised machine learning methods"_ at Uppsala University and in collaboration with Volvo Cars. The model is based on the OPN from the paper "Unsupervised Representation Learning by Sorting Sequence" (Lee, Huang, Singh & Yang, 2017).

The original code, based on the Caffe framework, can be found here: [https://github.com/HsinYingLee/OPN/tree/master](url)
   
## Requirements & Dependencies
Requirements for PyTorch, Azure Blob Storage and a few other libraries noted in the "Importing" section. Python 3.10.13 was used. 

## Models and Training Data
### Training Data
[UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) – Action Recognition Data Set, currently stored in Azure Blob Storage.

### Models
_final model to be posted here_

## Training
Since the data is loaded from Azure Blob Storage, automatically running the code and training the model should be pretty much plug-and-play. The order of execution is:
1. _data_prep.ipynb_ to set up the data
2. _main.py_ to run the training

## Testing
_more details to be added_

