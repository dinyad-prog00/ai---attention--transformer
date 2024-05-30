import tensorflow as tf
from keras.layers import Embedding, Layer
import numpy as np


class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, sequence_length, d_model, n=10000, fixed_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        if fixed_weights:
            word_embedding_matrix = self.positional_encoding(
                vocab_size, d_model)
            self.embedding = Embedding(
                input_dim=vocab_size, output_dim=d_model,
                weights=[word_embedding_matrix],
                trainable=False
            )
        else:
            self.embedding = Embedding(vocab_size, d_model, mask_zero=True)

        self.pos_encoding = self.positional_encoding(
            sequence_length, d_model, n)

    def positional_encoding(self, seq_len, d_model, n=10000):
        depth = d_model//2

        positions = np.arange(seq_len)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (n**depths)
        angle_rads = positions * angle_rates

        # Compute the sine and cosine components
        sin_vals = np.sin(angle_rads)
        cos_vals = np.cos(angle_rads)

        # Interleave sine and cosine values
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = sin_vals
        pos_encoding[:, 1::2] = cos_vals

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return self.embedding(x) + self.pos_encoding
