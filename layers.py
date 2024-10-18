import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        """
        Multi-head self-attention layer.

        Args:
        - hidden_size: Dimension of the hidden layers.
        - num_attention_heads: Number of attention heads.
        - dropout_prob: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.attention_head_size * num_attention_heads

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        query_layer = self.query(hidden_states).view(batch_size, seq_length, self.num_attention_heads,
                                                     self.attention_head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(batch_size, seq_length, self.num_attention_heads,
                                                 self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, seq_length, self.num_attention_heads,
                                                     self.attention_head_size).transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Ensure attention_mask has shape [batch_size, 1, 1, seq_length]
            if attention_mask.dim() == 2:  # If the mask has shape [batch_size, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Expand to [batch_size, 1, 1, seq_length]

            attention_scores = attention_scores + (attention_mask * -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2).reshape(batch_size, seq_length,
                                                                                           self.all_head_size)
        attention_output = self.out_proj(context_layer)
        return attention_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        """
        Feed-forward network.

        Args:
        - hidden_size: Size of the hidden layer.
        - intermediate_size: Size of the intermediate layer.
        - dropout_prob: Dropout probability.
        """
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob):
        """
        A single Transformer layer.

        Args:
        - hidden_size: Size of the hidden layer.
        - num_attention_heads: Number of attention heads.
        - intermediate_size: Size of the intermediate (feed-forward) layer.
        - dropout_prob: Dropout probability.
        """
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size, dropout_prob)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout1(attention_output)
        hidden_states = self.layer_norm1(hidden_states + attention_output)

        ffn_output = self.ffn(hidden_states)
        ffn_output = self.dropout2(ffn_output)
        hidden_states = self.layer_norm2(hidden_states + ffn_output)

        return hidden_states