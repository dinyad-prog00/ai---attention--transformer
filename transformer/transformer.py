import tensorflow as tf

from .decoder import Decoder
from .encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, input_vocab_size, output_vocab_size,sequence_length, h, d_k, d_v, d_model, d_ff, n=10000, fixed_weights=False, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(num_layers, input_vocab_size, sequence_length,
                               h, d_k, d_v, d_model, d_ff, n, fixed_weights, dropout_rate)
        self.decoder = Decoder(num_layers, output_vocab_size, sequence_length,
                               h, d_k, d_v, d_model, d_ff, n, fixed_weights, dropout_rate)
        self.linear = tf.keras.layers.Dense(output_vocab_size)

    def call(self, inputs):
        values, x  = inputs
        values = self.encoder(values)
        decoced = self.decoder(x, values)
        out = self.linear(decoced)
        return out
