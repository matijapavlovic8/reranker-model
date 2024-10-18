# Custom DistilBERT for Reranking

This project aims to develop a custom DistilBERT model for reranking tasks using the MS MARCO dataset. The workflow involves two main stages:
1. **Pretraining a custom DistilBERT model** using a BERT base uncased model as the teacher network for Masked Language Modeling (MLM).
2. **Fine-tuning the pretrained DistilBERT model** for reranking tasks using triplet loss on the MS MARCO dataset.

## Table of Contents

- [Overview](#overview)
- [Pretraining](#pretraining)
- [Fine-tuning](#fine-tuning)
- [Dependencies](#dependencies)


## Overview

The project workflow consists of the following steps:

1. **MLM Pretraining**: A custom DistilBERT model is pretrained using Masked Language Modeling, with BERT base uncased as the teacher network. The goal of this stage is to transfer the knowledge from BERT to a smaller and more efficient model.
2. **Reranking Fine-tuning**: The pretrained DistilBERT model is further fine-tuned on the MS MARCO dataset for reranking tasks using triplet loss. The goal is to train the model to distinguish relevant passages from irrelevant ones based on a given query.

## Pretraining

Pretraining the custom DistilBERT model involves using the BERT base uncased model as a teacher network for knowledge distillation. The MLM objective is employed to mask random tokens in the input and train the model to predict these masked tokens.

Steps:
1. Load BERT base uncased as the teacher model.
2. Train the DistilBERT model with the MLM objective using a dataset like Wikipedia or BookCorpus.
3. Perform knowledge distillation using the teacher model's logits to guide the learning process of DistilBERT.

## Fine-tuning

Fine-tuning involves training the pretrained DistilBERT model for a reranking task using the MS MARCO dataset and triplet loss. The MS MARCO dataset contains queries and corresponding relevant and non-relevant passages.

Steps:
1. Prepare the dataset by creating triplets consisting of a query (anchor), a relevant passage (positive), and a non-relevant passage (negative).
2. Use triplet loss to fine-tune the model, aiming to minimize the distance between the query and relevant passage embeddings while maximizing the distance from the non-relevant passage embeddings.
3. Evaluate the model's performance on a validation set to monitor its effectiveness.

## Dependencies

The project uses the following dependencies:
- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- NumPy