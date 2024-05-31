import tensorflow as tf

from .multi_head_attention_test import MultiHeadAttention


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, values):
        attn_output = self.mha(
            q=x,
            k=values,
            v=values)

        x = self.add([x, attn_output])
        x = self.layer_norm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            q=x,
            v=x,
            k=x)
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            q=x,
            v=x,
            k=x)
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
