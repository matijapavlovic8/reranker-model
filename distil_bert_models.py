import torch
import torch.nn as nn

from layers import TransformerLayer


class DistilBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512):
        super(DistilBERT, self).__init__()
        self.device = None
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_hidden_layers)
        ])

    def to(self, *args, **kwargs):
        # Override the .to() method to update the device attribute
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self

    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class DistilBERTForMLM(DistilBERT):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512):
        super(DistilBERTForMLM, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings
        )

        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = super().forward(input_ids, attention_mask)

        mlm_logits = self.mlm_head(self.activation(self.layer_norm(hidden_states)))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_logits = mlm_logits.view(-1, mlm_logits.size(-1))
            labels = labels.view(-1)
            mlm_loss = loss_fct(mlm_logits, labels)
            return mlm_loss, mlm_logits

        return mlm_logits

    def save_pretrained(self, save_directory):
        torch.save(self.state_dict(), f"{save_directory}/mlm_model.pt")


class DistilBERTForReranking(DistilBERT):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1, max_position_embeddings=512):
        super(DistilBERTForReranking, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob,
            max_position_embeddings=max_position_embeddings
        )

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = super().forward(input_ids, attention_mask)

        # Use the hidden state of the [CLS] token (first token) for sequence classification
        cls_token_state = hidden_states[:, 0, :]

        # Compute the relevance score
        scores = self.classifier(cls_token_state)
        return scores

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        model = cls()
        model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))

        return model

    def save_pretrained(self, save_directory):
        torch.save(self.state_dict(), save_directory)
