import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Construct the embeddings.

    Args:
        vocab_size (int): size of the vocabulary.
        d_model (int): size of the embedding vector.

    Returns:
        torch.Tensor: the embedding matrix.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch_size, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # mulitplicative parameter
        self.beta = nn.Parameter(torch.zeros(features))  # additive parameter
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RMSNorm(nn.Module):
    def __init(self, features: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features)) # mulitplicative parameter
        self.beta = nn.Parameter(torch.zerso(features)) # additive parameter
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        return self.gamma * (x) / (rms + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)  # W_1 and b_1
        self.linear2 = nn.Linear(d_ff, d_model)  # W_2 and b_2
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # W_q
        self.w_k = nn.Linear(d_model, d_model)  # W_k
        self.w_v = nn.Linear(d_model, d_model)  # W_v
        self.w_o = nn.Linear(d_model, d_model)  # W_o

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_score = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        x = self.w_o(x)  # (batch_size, seq_len, d_model)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
    ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionBlock(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        x = self.proj(x)
        return torch.log_softmax(x, dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        tgt_embed: Embeddings,
        src_pos_encod: PositionalEncoding,
        tgt_pos_encod: PositionalEncoding,
        proj: ProjectionBlock,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_encod = src_pos_encod
        self.tgt_pos_encod = tgt_pos_encod
        self.projection = proj

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos_encod(self.src_embed(src)), src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(
            self.tgt_pos_encod(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask
        )

    def projection(self, x):
        return self.projection(x)
