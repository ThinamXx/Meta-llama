# This is the implementation of the LLAMA2 model from the
# paper: Llama2: Open Foundation and Fine-Tuned Chat Models

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # given by tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-8
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.d_model = args.dim
        self.embed = nn.Embedding(args.vocab_size, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.eps = args.norm_eps
        self.weight = nn.Parameter(torch.ones(args.dim))  # mulitplicative parameter

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


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
        args: ModelArgs,
    ):
        super().__init__()

        d_ff = 4 * args.dim
        d_ff = int(2 * d_ff / 3)
        if args.ffn_dim_multiplier is not None:
            d_ff = int(d_ff * args.ffn_dim_multiplier)
        # make SwiGLU hidden layer size multiple of large power of 2
        d_ff = args.multiple_of * ((d_ff + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, d_ff, bias=False)

    def forward(self, x):
        swish = F.silu(self.w1(x))  # (batch_size, seq_len, d_ff)
        x_v = self.w3(x)  # (batch_size, seq_len, d_ff)
        x = swish * x_v  # (batch_size, seq_len, d_ff)
        x = self.w2(x)  # (batch_size, seq_len, d_model)
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


class GroupedQueryAttentionLLAMA(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()
        self.n_heads_q = args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim
        ).cuda()
        self.cache_v = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_heads * head_dim)
        query = self.wq(x)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_kv_heads * head_dim)
        key = self.wk(x)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_kv_heads * head_dim)
        value = self.wv(x)

        # (batch_size, seq_len, n_heads * head_dim) --> (batch_size, seq_len, n_heads, head_dim)
        query = query.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (batch_size, seq_len, n_kv_heads * head_dim) --> (batch_size, seq_len, n_kv_heads, head_dim)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (batch_size, seq_len, n_kv_heads * head_dim) --> (batch_size, seq_len, n_kv_heads, head_dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary positional embeddings to query and key
        # (batch_size, seq_len, n_heads, head_dim) --> (batch_size, seq_len, n_heads, head_dim)
        query = apply_rotary_pos_emb(
            query, freqs_complex=freqs_complex, device=x.device
        )
        # (batch_size, seq_len, n_kv_heads, head_dim) --> (batch_size, seq_len, n_kv_heads, head_dim)
        key = apply_rotary_pos_emb(key, freqs_complex=freqs_complex, device=x.device)

        self.cache_k = self.cache_k.to(query)
        self.cache_v = self.cache_v.to(query)

        # update the cache with the new key and value
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = key
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = value

        # (batch_size, seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]

        # repeat the keys and values n_rep times
        # (batch_size, seq_len_kv, n_kv_heads, head_dim) --> (batch_size, seq_len_kv, n_heads, head_dim)
        keys = repeat_kv_heads(keys, self.n_rep)
        values = repeat_kv_heads(values, self.n_rep)

        # (batch_size, seq_len, n_heads, head_dim) --> (batch_size, n_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        # (batch_size, seq_len_kv, n_heads, head_dim) --> (batch_size, n_heads, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        # (batch_size, seq_len_kv, n_heads, head_dim) --> (batch_size, n_heads, seq_len_kv, head_dim)
        values = values.transpose(1, 2)

        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len_kv) --> (batch_size, n_heads, seq_len, seq_len_kv)
        attention_scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        # (batch_size, n_heads, seq_len, seq_len_kv) --> (batch_size, n_heads, seq_len, seq_len_kv)
        attention_scores = F.softmax(attention_scores.float(), dim=-1).type_as(query)
        # (batch_size, n_heads, seq_len, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, head_dim) --> (batch_size, n_heads, seq_len, head_dim)
        attention = torch.matmul(attention_scores, values)
        # (batch_size, n_heads, seq_len, head_dim) --> (batch_size, seq_len, n_heads, head_dim)
        output = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (batch_size, seq_len, n_heads * head_dim) --> (batch_size, seq_len, d_model)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()
        self.attention = GroupedQueryAttentionLLAMA(args)
        self.feed_forward = FeedForwardBlock(args)

        # normalization before attention block and feed forward block
        self.attention_norm = RMSNorm(args)
        self.ffn_norm = RMSNorm(args)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        h = x + self.attention.forward(
            self.attention_norm(x),
            freqs_complex=freqs_complex,
            start_pos=start_pos,
            mask=mask,
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()

        assert args.vocab_size != -1, "Vocab_size must be specified"
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # token embedding
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # N x encoder layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerBlock(args))

        # rms normalization
        self.norm = RMSNorm(args)

        # projection layer
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_freqs(
            args.dim // args.n_heads, args.max_seq_len * 2, device=args.device
        )

    def forward(self, x: torch.Tensor, start_pos, mask: Optional[torch.Tensor]):
        _, seq_len = x.shape
        assert seq_len == 1, "Only one token can be generated at a time"

        # (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        h = self.tok_embeddings(x)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output
