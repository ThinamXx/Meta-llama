from llama.transformer import (
    Embeddings,
    PositionalEncoding,
    Encoder,
    EncoderBlock,
    Decoder,
    DecoderBlock,
    MultiHeadAttention,
    FeedForwardBlock,
    Transformer,
    ProjectionBlock,
)

import torch.nn as nn


# build transformer model
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
    # embedding layers
    src_embed = Embeddings(src_vocab_size, d_model)
    tgt_embed = Embeddings(tgt_vocab_size, d_model)

    # positional encoding
    src_pos_encod = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_encod = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder and decoder
    encoder = Encoder(
        d_model,
        nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    MultiHeadAttention(d_model, n_heads, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
    )
    decoder = Decoder(
        d_model,
        nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    MultiHeadAttention(d_model, n_heads, dropout),
                    MultiHeadAttention(d_model, n_heads, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
    )

    projection_layer = ProjectionBlock(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos_encod,
        tgt_pos_encod,
        projection_layer,
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer