# This is the implementation of the original LLAMA model
# from the paper: LLAMA: Open and Efficient Foundation Language Models

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def precompute_theta_pos_freqs(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    # mentioned in 3.2.2 of rotary positional embeddings paper
    assert head_dim % 2 == 0, "head dimension must be even"

    # (head_dim / 2)
    theta_num = torch.arange(0, head_dim, 2).float()
    theta_freqs = 1.0 / theta ** (theta_num / head_dim).to(device)

    m = torch.arange(seq_len, device=device)  # (seq_len)
    m_theta = torch.outer(m, theta_freqs).float()  # (seq_len, head_dim / 2)
    freqs_complex = torch.polar(
        torch.ones_like(m_theta), m_theta
    )  # (seq_len, head_dim / 2)

    return freqs_complex


def apply_rotary_pos_emb(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch_size, seq_len, head, head / 2)
    x_transform = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head, head / 2) --> (1, seq_len, 1, head / 2)
    freqs_complex_transform = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch_size, seq_len, head, head / 2) * (1, seq_len, 1, head / 2) --> (batch_size, seq_len, head, head / 2
    x_rotated = x_transform * freqs_complex_transform
    x_out = torch.view_as_real(x_rotated)  # (batch_size, seq_len, head, head / 2, 2)
    # (batch_size, seq_len, head, head / 2, 2) --> (batch_size, seq_len, head, head / 2 * 2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv_heads(x: torch.Tensor, reps: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if reps == 1:
        return x
    return (
        # (batch_size, seq_len_kv, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # (batch_size, seq_len_kv, n_kv_heads, reps, head_dim) --> (batch_size, seq_len_kv, n_kv_heads * reps, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, reps, head_dim).reshape(
            batch_size, seq_len, n_kv_heads * reps, head_dim
        )
    )


class Embeddings(nn.Module):
    """Construct the embeddings.

    Args:
        vocab_size (int): size of the vocabulary.
        d_model (int): size of the embedding vector.

    Returns:
        torch.Tensor: the embedding matrix.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


class RMSNorm(nn.Module):
    def __init(self, features: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))  # mulitplicative parameter
        self.beta = nn.Parameter(torch.zerso(features))  # additive parameter

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        return self.gamma * (x) / (rms + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    """Feed forward block for LLAMA model.

    Args:
        d_model (int): input dimension of model
        d_ff (int): hidden dimension of feed forward block
        ffn_dim_multiplier (int): custom multiplier for hidden dimension of feed forward block
        multiple_of (int): value to make hidden dimension of feed forward block multiple of
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        ffn_dim_multiplier: Optional[int],
        multiple_of: int,
    ):
        d_ff = 4 * d_model
        d_ff = int(2 * d_ff / 3)
        if ffn_dim_multiplier is not None:
            d_ff = d_ff * ffn_dim_multiplier
        # make SwiGLU hidden layer size multiple of large power of 2
        d_ff = multiple_of * ((d_ff * multiple_of - 1) // multiple_of)

        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        swish = F.silu(self.w_1(x))  # (batch_size, seq_len, d_ff)
        x_v = self.w_3(x)  # (batch_size, seq_len, d_ff)
        x = swish * x_v  # (batch_size, seq_len, d_ff)
        x = self.w_2(x)  # (batch_size, seq_len, d_model)
        return x


class MultiHeadAttentionLLAMA(nn.Module):
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

    def forward(self, q, k, v, mask, freqs_complex: torch.Tensor):
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # using rotary positional embeddings to rotate query and key
        # (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = apply_rotary_pos_emb(
            query, freqs_complex=freqs_complex, device=query.device
        ).transpose(1, 2)
        key = apply_rotary_pos_emb(
            key, freqs_complex=freqs_complex, device=key.device
        ).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionLLAMA.attention(
            query, key, value, mask, self.dropout
        )

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        x = self.w_o(x)  # (batch_size, seq_len, d_model)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        attention_block: MultiHeadAttentionLLAMA,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block

        # normalization before attention block and feed forward block
        self.attention_block_norm = RMSNorm(features, eps=1e-5)
        self.feed_forward_block_norm = RMSNorm(features, eps=1e-5)

    def forward(self, x, mask, freqs_complex):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        h = x + self.attention_block.forward(
            self.attention_block_norm(x),
            self.attention_block_norm(x),
            self.attention_block_norm(x),
            mask,
            freqs_complex,
        )
        x = h + self.feed_forward_block.forward(self.feed_forward_block_norm(h))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,  # hidden states
        vocab_size: int,
        seq_len: int,
        norm_eps: int = 1e-8,
        N: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.eps = norm_eps
        self.n_layers = N
        self.n_heads = n_heads
        self.head_dim = self.d_model // self.n_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier

        # token embedding
        self.embed = Embeddings(self.vocab_size, self.d_model)

        # rms normalization
        self.norm = RMSNorm(self.d_model, eps=self.eps)

        # N x encoder layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerBlock(
                    self.d_model,
                    MultiHeadAttentionLLAMA(self.d_model, self.head_dim, dropout),
                    FeedForwardBlock(
                        self.d_model,
                        self.d_ff,
                        self.ffn_dim_multiplier,
                        self.multiple_of,
                    ),
                    dropout,
                )
            )

        # projection layer
        self.out = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_freqs(self.head_dim, self.seq_len * 2)

    def forward(self, x: torch.Tensor, mask, start_pos: int = 0):
        _, seq_len = x.shape

        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        h = self.embed(x)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, mask, freqs_complex)

        h = self.norm(h)
        out = self.out(h).float()
        return out
