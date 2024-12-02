import random

import torch
from transformers import BertTokenizer
from datasets import load_dataset
from custom_datasets import TripletRerankingDataset
from distil_bert_models import DistilBERTForReranking
from utils import train_triplet_model


def main():
    tokenizer = BertTokenizer.from_pretrained("models/pretrained_model")
    distilbert_model = DistilBERTForReranking.from_pretrained("models/pretrained_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distilbert_model.to(device)

    dataset = load_dataset('ms_marco', 'v2.1', split='train')
    validation_dataset = load_dataset('ms_marco', 'v2.1', split='validation')

    queries = []
    relevant_docs = []
    irrelevant_docs = []

    skipped_queries = 0
    total_queries = 0

    for item in dataset:
        query = item['query']
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        total_queries += 1

        relevant_doc = None
        for i in range(len(passages)):
            if is_selected[i] == 1:
                relevant_doc = passages[i]
                break

        if relevant_doc is None:
            skipped_queries += 1
            continue

        negative_docs = [passages[i] for i in range(len(passages)) if is_selected[i] == 0]

        if not negative_docs:
            skipped_queries += 1
            continue

        negative_doc = random.choice(negative_docs)

        queries.append(query)
        relevant_docs.append(relevant_doc)
        irrelevant_docs.append(negative_doc)

    skipped_percentage = (skipped_queries / total_queries) * 100
    print(f"Skipped {skipped_queries} queries out of {total_queries} total queries.")
    print(f"Percentage of skipped queries: {skipped_percentage:.2f}%")

    print(f"Training data size: {len(queries)} queries")
    triplet_dataset = TripletRerankingDataset(queries, relevant_docs, irrelevant_docs, tokenizer)

    val_queries = []
    val_relevant_docs = []
    val_irrelevant_docs = []

    skipped_val_queries = 0
    total_val_queries = 0

    for item in validation_dataset:
        query = item['query']
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        total_val_queries += 1

        relevant_doc = None
        for i in range(len(passages)):
            if is_selected[i] == 1:
                relevant_doc = passages[i]
                break

        if relevant_doc is None:
            skipped_val_queries += 1
            continue

        negative_docs = [passages[i] for i in range(len(passages)) if is_selected[i] == 0]

        if not negative_docs:
            skipped_val_queries += 1
            continue

        negative_doc = random.choice(negative_docs)

        val_queries.append(query)
        val_relevant_docs.append(relevant_doc)
        val_irrelevant_docs.append(negative_doc)

    skipped_val_percentage = (skipped_val_queries / total_val_queries) * 100
    print(f"Skipped {skipped_val_queries} validation queries out of {total_val_queries} total validation queries.")
    print(f"Percentage of skipped validation queries: {skipped_val_percentage:.2f}%")

    print(f"Validation data size: {len(val_queries)} queries")
    val_triplet_dataset = TripletRerankingDataset(val_queries, val_relevant_docs, val_irrelevant_docs, tokenizer)

    train_triplet_model(distilbert_model, triplet_dataset, val_triplet_dataset, num_epochs=3, batch_size=2)


if __name__ == "__main__":
    main()

