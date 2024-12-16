import os
from typing import cast, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = F.relu(margin + neg_sim - pos_sim).mean()
    return loss

def train_triplet_model(model, train_dataset, val_dataset, num_epochs=3, batch_size=2):
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            anchor_input_ids = batch[0].to(model.device)
            anchor_attention_mask = batch[1].to(model.device)
            positive_input_ids = batch[2].to(model.device)
            positive_attention_mask = batch[3].to(model.device)
            negative_input_ids = batch[4].to(model.device)
            negative_attention_mask = batch[5].to(model.device)

            anchor_last_hidden_state, _ = model(anchor_input_ids, attention_mask=anchor_attention_mask)
            positive_last_hidden_state, _ = model(positive_input_ids, attention_mask=positive_attention_mask)
            negative_last_hidden_state, _ = model(negative_input_ids, attention_mask=negative_attention_mask)

            anchor_embeddings = anchor_last_hidden_state
            positive_embeddings = positive_last_hidden_state
            negative_embeddings = negative_last_hidden_state

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (step + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}, Step {step + 1}, Training Loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                anchor_input_ids = batch[0].to(model.device)
                anchor_attention_mask = batch[1].to(model.device)
                positive_input_ids = batch[2].to(model.device)
                positive_attention_mask = batch[3].to(model.device)
                negative_input_ids = batch[4].to(model.device)
                negative_attention_mask = batch[5].to(model.device)

                anchor_last_hidden_state, _ = model(anchor_input_ids, attention_mask=anchor_attention_mask)
                positive_last_hidden_state, _ = model(positive_input_ids, attention_mask=positive_attention_mask)
                negative_last_hidden_state, _ = model(negative_input_ids, attention_mask=negative_attention_mask)

                anchor_embeddings = anchor_last_hidden_state
                positive_embeddings = positive_last_hidden_state
                negative_embeddings = negative_last_hidden_state

                val_loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss}")

def train_distillation(bert_model, distilbert_model, train_dataset, val_dataset, tokenizer,
                       num_epochs: int = 3, batch_size: int = 2, alpha: float = 0.5,
                       temperature: float = 2.0) -> None:
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    distilbert_model.train()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(distilbert_model.parameters(), lr=2e-5)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        total_loss = 0
        total_steps = len(train_dataloader)

        for step, batch in enumerate(train_dataloader):
            batch = cast(Dict[str, torch.Tensor], batch)
            input_ids = batch['input_ids'].to(distilbert_model.device)
            attention_mask = batch['attention_mask'].to(distilbert_model.device)
            labels = batch['labels'].to(distilbert_model.device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = bert_model(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = distilbert_model(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs

            mlm_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

            distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(student_logits / temperature, dim=-1),
                nn.functional.softmax(teacher_logits / temperature, dim=-1)
            )

            loss = alpha * distillation_loss * (temperature ** 2) + (1.0 - alpha) * mlm_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 5 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{total_steps}, Training Loss: {avg_loss:.4f}")

        avg_loss = total_loss / total_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        distilbert_model.eval()
        total_val_loss = 0
        val_steps = len(val_dataloader)
        with torch.no_grad():
            for val_step, val_batch in enumerate(val_dataloader):
                val_batch = cast(Dict[str, torch.Tensor], val_batch)
                input_ids = val_batch['input_ids'].to(distilbert_model.device)
                attention_mask = val_batch['attention_mask'].to(distilbert_model.device)
                labels = val_batch['labels'].to(distilbert_model.device)

                teacher_outputs = bert_model(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

                student_outputs = distilbert_model(input_ids, attention_mask=attention_mask)
                student_logits = student_outputs

                mlm_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
                distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                    nn.functional.log_softmax(student_logits / temperature, dim=-1),
                    nn.functional.softmax(teacher_logits / temperature, dim=-1)
                )

                val_loss = alpha * distillation_loss * (temperature ** 2) + (1.0 - alpha) * mlm_loss
                total_val_loss += val_loss.item()

                if (val_step + 1) % 5 == 0:
                    avg_val_loss = total_val_loss / (val_step + 1)
                    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Step {val_step + 1}/{val_steps}, Validation Loss: {avg_val_loss:.4f}")

        avg_val_loss = total_val_loss / val_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")
    model_save_path = "models/pretrained_model"
    os.makedirs(model_save_path, exist_ok=True)
    distilbert_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")