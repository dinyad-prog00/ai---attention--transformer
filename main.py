from numpy import random

from multi_head_attention_test import MultiHeadAttention

input_seq_length = 5  # Maximum length of the input sequence
d_k = 4  # Dimensionality of the linearly projected queries and keys
d_v = 4  # Dimensionality of the linearly projected values
d_model = 8  # Dimensionality of the model sub-layers' outputs
h = 2  # Number of self-attention heads
batch_size = 4  # Batch size from the training process

q = random.random((batch_size, input_seq_length, d_k))
k = random.random((batch_size, input_seq_length, d_k))
v = random.random((batch_size, input_seq_length, d_v))

mh_attention = MultiHeadAttention(h, d_k, d_v, d_model)
o = mh_attention(q, k, v)
print(o.shape)
print(o)
