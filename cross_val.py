import torch
from transformers import BertForMaskedLM, BertTokenizer
from datasets import load_dataset
from custom_datasets import TextDataset
from distil_bert_models import DistilBERTForMLM
from utils import train_distillation
import json
from itertools import product


def main(config, train_hyperparams):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    distilbert_model = DistilBERTForMLM(
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        dropout_prob=config['dropout_prob'],
        max_position_embeddings=config['max_position_embeddings']
    )

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
        num_epochs=train_hyperparams['num_epochs'],
        batch_size=train_hyperparams['batch_size'],
        alpha=train_hyperparams['alpha'],
        temperature=train_hyperparams['temperature']
    )

    return avg_loss, avg_val_loss


def run_cross_validation(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)

    models = data.get("models", [])
    train_hyperparams = data.get("training_hyperparams", {})

    assert isinstance(models, list), "Models must be a list of dictionaries"
    assert isinstance(train_hyperparams, dict), "Training hyperparams must be a dictionary"

    param_combinations = list(product(
        train_hyperparams['num_epochs'],
        train_hyperparams['batch_size'],
        train_hyperparams['alpha'],
        train_hyperparams['temperature']
    ))

    results = {}

    for model_config in models:
        model_id = model_config['model_id']
        results[model_id] = []

        print(f"Running Model ID: {model_id}")
        for params in param_combinations:
            param_dict = {
                'num_epochs': params[0],
                'batch_size': params[1],
                'alpha': params[2],
                'temperature': params[3]
            }

            print(f"Training with params: {param_dict}")
            avg_loss, avg_val_loss = main(model_config, param_dict)

            results[model_id].append({
                'params': param_dict,
                'avg_loss': avg_loss,
                'avg_val_loss': avg_val_loss
            })

    with open('cross_validation_results.txt', 'w') as f:
        for model_id, model_results in results.items():
            f.write(f"Model ID: {model_id}\n")
            for result in model_results:
                f.write(
                    f"Params: {result['params']}, Avg Loss: {result['avg_loss']:.4f}, Avg Val Loss: {result['avg_val_loss']:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    run_cross_validation('config.json')
