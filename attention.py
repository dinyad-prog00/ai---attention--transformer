from tensorflow import  matmul, math, cast, float32
from keras.layers import Layer
from keras.backend import softmax


class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, q, k, v, d_k, mask=None):
        scores = matmul(q, k, transpose_b=True)/math.rsqrt(cast(d_k,float32))
        
        if mask is not None:
            scores += -1e9 * mask
        
        return matmul(softmax(scores),v)