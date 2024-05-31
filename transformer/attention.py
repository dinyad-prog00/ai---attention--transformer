from tensorflow import  matmul, math, cast, float32,logical_not
from keras.layers import Layer
import tensorflow as tf


class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, q, k, v, d_k, mask=None):
        scores = matmul(q, k, transpose_b=True)/math.rsqrt(cast(d_k,float32))
        
        if mask is not None:
            mask= logical_not(mask)
            mask= cast(mask,dtype=scores.dtype)
            scores += -1e9 * mask
        
        return matmul(tf.keras.backend.softmax(scores),v)