from tensorflow import reshape, shape, transpose, ones, cumsum, greater_equal
from keras.layers import Layer, Dense

from .attention import ScaledDotProductAttention


class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, masked=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = ScaledDotProductAttention()
        self.num_heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.masked = masked
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

    def _compute_causal_mask(self, q, v):
        q_seq_length = shape(q)[1]
        v_seq_length = q_seq_length if v is None else shape(v)[1]
        ones_mask = ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = cumsum(ones_mask, axis=-2)
        col_index = cumsum(ones_mask, axis=-1)
        return greater_equal(row_index, col_index)

    def call(self, q, k, v):
        q_reshaped = self._heads_split(self.W_q(q), self.num_heads)

        k_reshaped = self._heads_split(self.W_k(k), self.num_heads)

        v_reshaped = self._heads_split(self.W_v(v), self.num_heads)

        mask = None
        if self.masked:
            mask = self._compute_causal_mask(q, v)

        o_reshaped = self.attention(
            q_reshaped, k_reshaped, v_reshaped, d_k=self.d_k, mask=mask)

        output = self._heads_contat(o_reshaped)

        return self.W_o(output)
