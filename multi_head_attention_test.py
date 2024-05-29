from tensorflow import  reshape, shape, transpose
from keras.layers import Layer, Dense

from attention import ScaledDotProductAttention

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v,d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = ScaledDotProductAttention()
        self.num_heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model  
        self.W_q = Dense(d_k)  
        self.W_k = Dense(d_k)  
        self.W_v = Dense(d_v)  
        self.W_o = Dense(d_model) 
        
    def _heads_split(self, x, h):
       # out shape: (batch_size, heads, seq_length, -1)
       x = reshape(x, shape=(shape(x)[0], shape(x)[1], h, -1))
       x = transpose(x, perm=(0, 2, 1, 3))
       return x
        
    def _heads_contat(self, x):
        # out shape: (batch_size, seq_length, d_k)
        x = transpose(x, perm=(0, 2, 1, 3))
        x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, q, k, v, mask=None):
        q_reshaped = self._heads_split(self.W_q(q), self.num_heads)
        
        k_reshaped = self._heads_split(self.W_k(k), self.num_heads)
       
        v_reshaped = self._heads_split(self.W_v(v), self.num_heads)
        
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        
        output = self._heads_contat(o_reshaped)
        
        return self.W_o(output)

