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

        anchor_inputs = self.tokenizer(
            text=query, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )
        positive_inputs = self.tokenizer(
            text=pos_doc, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )
        negative_inputs = self.tokenizer(
            text=neg_doc, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )

        return (
            anchor_inputs['input_ids'].squeeze(0), anchor_inputs['attention_mask'].squeeze(0),
            positive_inputs['input_ids'].squeeze(0), positive_inputs['attention_mask'].squeeze(0),
            negative_inputs['input_ids'].squeeze(0), negative_inputs['attention_mask'].squeeze(0)
        )
