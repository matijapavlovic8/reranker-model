import random

import torch
from transformers import BertTokenizer
from datasets import load_dataset
from custom_datasets import TripletRerankingDataset  # Assuming you have this class
from distil_bert_models import DistilBERTForReranking
from utils import train_triplet_model  # Assuming this function is defined


def main():
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the pretrained model weights
    # distilbert_model = DistilBERTForReranking.from_pretrained("models/pretrained_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # distilbert_model.to(device)

    # Load the dataset
    dataset = load_dataset('ms_marco', 'v2.1', split='train')

    queries = []
    relevant_docs = []
    irrelevant_docs = []

    for item in dataset:
        query = item['query']  # Get the query

        # Extract the relevant documents
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']

        # Get the first relevant document (if any)
        relevant_doc = next((passages[i] for i in range(len(passages)) if is_selected[i] == 1), None)

        # Generate a negative sample (irrelevant document)
        negative_docs = [passages[i] for i in range(len(passages)) if is_selected[i] == 0]

        # Choose a random negative document
        negative_doc = random.choice(negative_docs) if negative_docs else None

        queries.append(query)
        relevant_docs.append(relevant_doc)
        irrelevant_docs.append(negative_doc)

    # Create the triplet dataset
    triplet_dataset = TripletRerankingDataset(queries, relevant_docs, irrelevant_docs, tokenizer)

    # Fine-tune the model
    # train_triplet_model(distilbert_model, triplet_dataset, num_epochs=3, batch_size=2)


if __name__ == "__main__":
    main()
