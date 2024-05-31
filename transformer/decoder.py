import tensorflow as tf
from .base import CausalSelfAttention, FeedForward, CrossAttention
from .positional_encoding import PositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            h=h, d_k=d_k, d_v=d_v, d_model=d_model)

        self.cross_attention = CrossAttention(
            h=h, d_k=d_k, d_v=d_v, d_model=d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

    def call(self, x, values):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, values=values)

        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n=10000, fixed_weights=False, dropout_rate=0.1,):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            d_model=d_model,
            n=n,
            fixed_weights=fixed_weights
        )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.decoder_layers = [
            DecoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, x,values):
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x,values)

        return x
