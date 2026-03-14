import math
from typing import override

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.model_dim: int = model_dim
        self.vocab_size: int = vocab_size
        self.embd: nn.Embedding = nn.Embedding(vocab_size, model_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embd(x) * math.sqrt(self.model_dim)


class PositionalEncoding(nn.Module):
    __slots__: tuple[str, ...] = ("model_dim", "seq_len", "dropout", "pe")

    def __init__(self, model_dim: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.model_dim: int = model_dim
        self.seq_len: int = seq_len
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.pe: torch.Tensor

        pe = torch.zeros(seq_len, model_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [10,512]->[1,10,512]
        self.register_buffer(
            "pe", pe
        )  # save positional encoding when the model is saved

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    __slots__: tuple[str, ...] = ("eps", "alpha", "bias")

    def __init__(self, features_size: int = 512, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.alpha: torch.Tensor = nn.Parameter(torch.ones(features_size))
        self.bias: torch.Tensor = nn.Parameter(torch.zeros(features_size))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(model_dim, d_ff)  # W2,b1
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear2: nn.Linear = nn.Linear(d_ff, model_dim)  # w2,b2

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, h: int, dropout: float) -> None:
        super().__init__()
        self.model_dim: int = model_dim
        self.h: int = h
        assert model_dim % h == 0, "model_dim is not divisible by h"
        self.d_k: int = model_dim // h
        self.wq: nn.Linear = nn.Linear(model_dim, model_dim)
        self.wk: nn.Linear = nn.Linear(model_dim, model_dim)
        self.wv: nn.Linear = nn.Linear(model_dim, model_dim)
        self.wo: nn.Linear = nn.Linear(model_dim, model_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.attention_scores: torch.Tensor

    @override
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        query: torch.Tensor = self.wq(q)
        key: torch.Tensor = self.wk(k)
        value: torch.Tensor = self.wv(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        )  # (batch, h, seq_len, d_k)-> #(batch, seq_len, h, d_k)
        return self.wo(x)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        dropout: nn.Dropout | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            _ = attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=1)  # head dimension
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score


# class ResidualConnection(nn.Module):
#     def __init__(self, dropout: float):
#         super().__init__()
#         self.dropout: nn.Dropout = nn.Dropout(dropout)
#         self.norm: LayerNormalization = LayerNormalization()
#
#     @override
#     def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
#         return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block: MultiHeadAttentionBlock = self_attention_block
        self.feed_forward_block: FeedForwardBlock = feed_forward_block

        self.norm1: LayerNormalization = LayerNormalization()
        self.norm2: LayerNormalization = LayerNormalization()

        self.dropout1: nn.Dropout = nn.Dropout(dropout)
        self.dropout2: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out: torch.Tensor = self.self_attention_block(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out: torch.Tensor = self.feed_forward_block(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block: MultiHeadAttentionBlock = self_attention_block
        self.feed_forward_block: FeedForwardBlock = feed_forward_block
        self.cross_attention_block: MultiHeadAttentionBlock = cross_attention_block

        self.norm1: LayerNormalization = LayerNormalization()
        self.norm2: LayerNormalization = LayerNormalization()
        self.norm3: LayerNormalization = LayerNormalization()

        self.dropout1: nn.Dropout = nn.Dropout(dropout)
        self.dropout2: nn.Dropout = nn.Dropout(dropout)
        self.dropout3: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(
        self,
        x: torch.Tensor,
        encoded: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        # self-attention
        x_residual: torch.Tensor = x
        x = self.norm1(x)
        attn_out: torch.Tensor = self.self_attention_block(x, x, x, tgt_mask)
        x = x_residual + self.dropout1(attn_out)

        # cross-attention
        x_residual = x
        x = self.norm2(x)
        attn_out = self.cross_attention_block(x, encoded, encoded, src_mask)
        x = x_residual + self.dropout2(attn_out)

        # feed-forward
        x_residual = x
        x = self.norm3(x)
        attn_out = self.feed_forward_block(x)
        x = x_residual + self.dropout3(attn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.norm: LayerNormalization = LayerNormalization()

    @override
    def forward(self, x, encoded, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoded, src_mask, tgt_mask)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.norm: LayerNormalization = LayerNormalization()

    @override
    def forward(self, x, mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_module: int, vocab_size: int):
        super().__init__()
        self.proj: nn.Linear = nn.Linear(d_module, vocab_size)

    @override
    def forward(self, x) -> torch.Tensor:
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embd: InputEmbedding,
        tgt_embd: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection: ProjectionLayer,
    ):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.src_embd: InputEmbedding = src_embd
        self.tgt_embd: InputEmbedding = tgt_embd
        self.src_pos: PositionalEncoding = src_pos
        self.tgt_pos: PositionalEncoding = tgt_pos
        self.projection: ProjectionLayer = projection

    def encode(self, src, src_mask) -> torch.Tensor:
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask) -> torch.Tensor:
        tgt = self.tgt_embd(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x) -> torch.Tensor:
        return self.projection(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    model_dim: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    src_embd = InputEmbedding(model_dim, src_vocab_size)
    tgt_embd = InputEmbedding(model_dim, tgt_vocab_size)
    src_pos = PositionalEncoding(model_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(model_dim, tgt_seq_len, dropout)

    encoder_blocks = []
    decoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(model_dim, h, dropout)
        ffw = FeedForwardBlock(model_dim, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(encoder_self_attention, ffw, dropout))

        ffw_2 = FeedForwardBlock(model_dim, d_ff, dropout)
        decoder_self_attention = MultiHeadAttentionBlock(model_dim, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(model_dim, h, dropout)
        decoder_blocks.append(
            DecoderBlock(
                decoder_self_attention, decoder_cross_attention, ffw_2, dropout
            )
        )
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(model_dim, tgt_vocab_size)
    transformer = Transformer(
        encoder,
        decoder,
        src_embd,
        tgt_embd,
        src_pos,
        tgt_pos,
        projection_layer,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            _ = nn.init.xavier_uniform_(p)
    return transformer
