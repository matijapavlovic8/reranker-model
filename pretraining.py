import torch
from transformers import BertForMaskedLM, BertTokenizer
from datasets import load_dataset

from custom_datasets import TextDataset
from distil_bert_models import DistilBERTForMLM
from utils import train_distillation


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    distilbert_model = DistilBERTForMLM()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    distilbert_model.to(device)

    dataset = load_dataset('wikicorpus', 'raw_en', split='train[:1%]')
    texts = dataset['text']

    tokenized_dataset = TextDataset(texts=texts, tokenizer=tokenizer)

    train_size = int(0.8 * len(tokenized_dataset))
    val_size = len(tokenized_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, val_size])

    avg_loss, avg_val_loss = train_distillation(
        bert_model=bert_model,
        distilbert_model=distilbert_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        num_epochs=3,
        batch_size=8,
        alpha=0.5,
        temperature=2.0
    )

    print(avg_loss, avg_val_loss)

if __name__ == "__main__":
    main()