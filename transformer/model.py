import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.drop_out = nn.Dropout(drop_out)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # making the [10,512]->[1,10,512]
        self.register_buffer(
            "pe", pe
        )  # save the positional encoding when the model is saved

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.drop_out(x)


class LayerNormalization(nn.Module):
    def __init__(self, features_size: int = 512, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features_size))
        self.bias = nn.Parameter(torch.zeros(features_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, drop_out: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W2,b1
        self.drop_out = nn.Dropout(drop_out)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2,b2

    def forward(self, x):
        return self.linear_2(self.drop_out(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (batch,h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.drop_out
        )
        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        )  # (batch, h, seq_len, d_k)-> #(batch,seq_len,h, d_k)
        return self.w_o(x)

    @staticmethod
    def attention(query, key, value, mask, drop_out: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=1)  # head dimension
        if drop_out is not None:
            attention_score = drop_out(attention_score)
        return (attention_score @ value), attention_score


class ResidualConnection(nn.Module):
    def __init__(self, drop_out: float):
        super().__init__()
        self.drop_out = nn.Dropout(drop_out)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.drop_out(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for i in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )  # wrap the forward into a one parameter function
        x = self.residual_connections[1](x, self.feed_forward_block)
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
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for i in range(3)]
        )

    def forward(self, output, encoded, src_mask, target_mask):
        output = self.residual_connection[0](
            output, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        output = self.residual_connection[1](
            output, lambda x: self.cross_attention_block(x, encoded, encoded, src_mask)
        )
        output = self.residual_connection[2](output, self.feed_forward_block)
        return output


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, output, encoded, src_mask, target_mask):
        for layer in self.layers:
            output = layer(output, encoded, src_mask, target_mask)
        return self.norm(output)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_module: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_module, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embedding: InputEmbedding,
        src_pos: PositionalEncoding,
        tar_pos: PositionalEncoding,
        projection: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embedding
        self.src_pos = src_pos
        self.tar_pos = tar_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tar, tar_mask):
        tar = self.tgt_embed(tar)
        tar = self.tar_pos(tar)
        return self.decoder(tar, encoder_output, src_mask, tar_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(
    src_vocab_size: int,
    tar_vocab_size: int,
    src_seq_len: int,
    tar_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    drop_out: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tar_embedding = InputEmbedding(d_model, tar_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, drop_out)
    tar_pos = PositionalEncoding(d_model, tar_seq_len, drop_out)

    encoder_blocks = []
    decoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, drop_out)
        ffw = FeedForwardBlock(d_model, d_ff, drop_out)
        encoder_blocks.append(EncoderBlock(encoder_self_attention, ffw, drop_out))

        ffw_2 = FeedForwardBlock(d_model, d_ff, drop_out)
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, drop_out)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, drop_out)
        decoder_blocks.append(
            DecoderBlock(
                decoder_self_attention, decoder_cross_attention, ffw_2, drop_out
            )
        )
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tar_vocab_size)
    transformer = Transformer(
        encoder,
        decoder,
        src_embedding,
        tar_embedding,
        src_pos,
        tar_pos,
        projection_layer,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
