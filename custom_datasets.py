from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class TripletRerankingDataset(Dataset):
    def __init__(self, queries, positive_docs, negative_docs, tokenizer):
        """
        Initialize the dataset for triplet loss training.

        Args:
        - queries: List of query texts (anchor).
        - positive_docs: List of documents relevant to each query.
        - negative_docs: List of documents not relevant to each query.
        - tokenizer: Tokenizer used for converting text to input tensors.
        """
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_doc = self.positive_docs[idx]
        neg_doc = self.negative_docs[idx]

        # Tokenizing inputs
        anchor_inputs = self.tokenizer(query, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        positive_inputs = self.tokenizer(pos_doc, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        negative_inputs = self.tokenizer(neg_doc, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        # Squeeze to get rid of extra dimensions
        anchor_input_ids = anchor_inputs['input_ids'].squeeze(0)
        anchor_attention_mask = anchor_inputs['attention_mask'].squeeze(0)
        positive_input_ids = positive_inputs['input_ids'].squeeze(0)
        positive_attention_mask = positive_inputs['attention_mask'].squeeze(0)
        negative_input_ids = negative_inputs['input_ids'].squeeze(0)
        negative_attention_mask = negative_inputs['attention_mask'].squeeze(0)

        return (
            anchor_input_ids, anchor_attention_mask,
            positive_input_ids, positive_attention_mask,
            negative_input_ids, negative_attention_mask
        )