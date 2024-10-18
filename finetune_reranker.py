import random

import torch
from transformers import BertTokenizer
from datasets import load_dataset
from custom_datasets import TripletRerankingDataset  # Assuming you have this class
from distil_bert_models import DistilBERTForReranking
from utils import train_triplet_model  # Assuming this function is defined


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # distilbert_model = DistilBERTForReranking.from_pretrained("models/pretrained_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # distilbert_model.to(device)

    dataset = load_dataset('ms_marco', 'v2.1', split='train')
    validation_dataset = load_dataset('ms_marco', 'v2.1', split='validation')
    queries = []
    relevant_docs = []
    irrelevant_docs = []

    for item in dataset:
        query = item['query']

        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        relevant_doc = next((passages[i] for i in range(len(passages)) if is_selected[i] == 1), None)

        negative_docs = [passages[i] for i in range(len(passages)) if is_selected[i] == 0]

        negative_doc = random.choice(negative_docs) if negative_docs else None

        queries.append(query)
        relevant_docs.append(relevant_doc)
        irrelevant_docs.append(negative_doc)

    triplet_dataset = TripletRerankingDataset(queries, relevant_docs, irrelevant_docs, tokenizer)

    val_queries = []
    val_relevant_docs = []
    val_irrelevant_docs = []

    for item in validation_dataset:
        query = item['query']

        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        relevant_doc = next((passages[i] for i in range(len(passages)) if is_selected[i] == 1), None)

        negative_docs = [passages[i] for i in range(len(passages)) if is_selected[i] == 0]
        negative_doc = random.choice(negative_docs) if negative_docs else None

        val_queries.append(query)
        val_relevant_docs.append(relevant_doc)
        val_irrelevant_docs.append(negative_doc)

    val_triplet_dataset = TripletRerankingDataset(val_queries, val_relevant_docs, val_irrelevant_docs, tokenizer)

    #train_triplet_model(distilbert_model, triplet_dataset, val_triplet_dataset, num_epochs=3, batch_size=2)


if __name__ == "__main__":
    main()
