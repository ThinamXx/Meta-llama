from model import (
    Embeddings,
    PositionalEncoding,
    Encoder,
    EncoderBlock,
    Decoder,
    DecoderBlock,
    MultiHeadAttention,
    FeedForwardBlock,
    Transformer,
)

import torch
import torch.nn as nn


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: 512,
    d_ff: 2048,
    src_seq_len: int,
    tgt_seq_len: int,
    dropout: float,
    n_layers: 8,
    n_heads: 8,
) -> Transformer:
    src_embed = Embeddings(src_vocab_size, d_model)
    tgt_embed = Embeddings(tgt_vocab_size, d_model)

    src_pos_encod = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_encod = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder = Encoder(
        nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttention(d_model, n_heads, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
    )
    
    decoder = Decoder(
        nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttention(d_model, n_heads, dropout), 
                    MultiHeadAttention(d_model, n_heads, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
    )
