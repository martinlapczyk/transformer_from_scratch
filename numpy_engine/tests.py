
import numpy as np
from attention import softmax, scaled_dot_product_attention, MultiHeadAttention
from positional_encoding import positional_encoding
from layers import layer_norm, feedforward
from transformer_block import TransformerBlock
def test_attention_shape():
    rng = np.random.default_rng(0)
    seq_len, d_k = 4, 8
    Q = rng.normal(size=(seq_len, d_k))
    K = rng.normal(size=(seq_len, d_k))
    V = rng.normal(size=(seq_len, d_k))

    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (seq_len, d_k), f"expected {(seq_len, d_k)}, got {out.shape}"
    assert weights.shape == (seq_len, seq_len), f"expected {(seq_len, seq_len)}, got {weights.shape}"
    print("test_attention_shape passed")


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    Q = K = V = rng.normal(size=(4, 8))
    _, weights = scaled_dot_product_attention(Q, K, V)
    row_sums = weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"row sums not 1: {row_sums}"
    print("test_weights_sum_to_one passed")


def test_masking_blocks_diagonal():
    rng = np.random.default_rng(0)
    seq_len = 4
    Q = K = V = rng.normal(size=(seq_len, 8))
    mask = 1 - np.eye(seq_len)
    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    diag = np.diagonal(weights, axis1=-2, axis2=-1)
    assert np.allclose(diag, 0.0, atol=1e-6), f"diagonal not ~0: {diag}"
    print("test_masking_blocks_diagonal passed")


def test_scaling_reduces_peakiness():
    rng = np.random.default_rng(1)
    seq_len, d_k = 4, 256
    Q = rng.normal(size=(seq_len, d_k))
    K = rng.normal(size=(seq_len, d_k))
    V = rng.normal(size=(seq_len, d_k))

    _, scaled_weights = scaled_dot_product_attention(Q, K, V)

    scores_unscaled = Q @ np.swapaxes(K, -1, -2)  # no /sqrt(d_k)
    unscaled_weights = softmax(scores_unscaled, axis=-1)

    scaled_max = scaled_weights.max(axis=-1).mean()
    unscaled_max = unscaled_weights.max(axis=-1).mean()

    assert unscaled_max > scaled_max, (
        f"expected unscaled ({unscaled_max:.4f}) to be peakier than scaled ({scaled_max:.4f})"
    )
    print("test_scaling_reduces_peakiness passed")

def test_multihead_shape_preserved():
    d_model, num_heads, seq_len, batch = 16, 4, 5, 2
    mha = MultiHeadAttention(d_model, num_heads, seed=0)
    x = np.random.default_rng(0).normal(size=(batch, seq_len, d_model))
    out, weights = mha.forward(x)
    assert out.shape == (batch, seq_len, d_model), f"got {out.shape}"
    print("test_multihead_shape_preserved passed")


def test_split_combine_heads_are_inverses():
    d_model, num_heads, seq_len, batch = 16, 4, 5, 2
    mha = MultiHeadAttention(d_model, num_heads, seed=0)
    x = np.random.default_rng(0).normal(size=(batch, seq_len, d_model))
    split = mha.split_heads(x)
    recombined = mha.combine_heads(split)
    assert np.allclose(x, recombined), "split_heads -> combine_heads did not round-trip"
    print("test_split_combine_heads_are_inverses passed")


def test_num_heads_must_divide_d_model():
    try:
        MultiHeadAttention(d_model=10, num_heads=3)  # 10 % 3 != 0
        assert False, "expected AssertionError for invalid num_heads"
    except AssertionError:
        print("test_num_heads_must_divide_d_model passed")

def test_positional_encoding_shape():
    seq_len, d_model = 10, 16
    pe = positional_encoding(seq_len, d_model)
    assert pe.shape == (seq_len, d_model), f"got {pe.shape}"
    print("test_positional_encoding_shape passed")


def test_positional_encoding_bounded():
    pe = positional_encoding(10, 16)
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0), "PE values should be in [-1, 1] (sin/cos range)"
    print("test_positional_encoding_bounded passed")


def test_positional_encoding_position_zero_is_predictable():
    pe = positional_encoding(seq_len=5, d_model=8)
    assert np.allclose(pe[0, 0::2], 0.0, atol=1e-6), "even dims at pos 0 should be ~0"
    assert np.allclose(pe[0, 1::2], 1.0, atol=1e-6), "odd dims at pos 0 should be ~1"
    print("test_positional_encoding_position_zero_is_predictable passed")

def test_layer_norm_output_stats():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(4, 16))  # arbitrary mean/std
    normed = layer_norm(x)
    row_means = normed.mean(axis=-1)
    row_stds = normed.std(axis=-1)
    assert np.allclose(row_means, 0.0, atol=1e-5), f"means not ~0: {row_means}"
    assert np.allclose(row_stds, 1.0, atol=1e-3), f"stds not ~1: {row_stds}"
    print("test_layer_norm_output_stats passed")


def test_layer_norm_handles_zero_variance():
    x = np.full((1, 8), 5.0)  # constant row -> std = 0
    normed = layer_norm(x)
    assert not np.any(np.isnan(normed)), "layer_norm produced NaN on zero-variance input"
    print("test_layer_norm_handles_zero_variance passed")


def test_feedforward_shape_and_relu():
    d_model, d_ff, seq_len = 8, 32, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=(seq_len, d_model))
    W1 = rng.normal(size=(d_model, d_ff))
    b1 = np.zeros(d_ff)
    W2 = rng.normal(size=(d_ff, d_model))
    b2 = np.zeros(d_model)

    out = feedforward(x, W1, b1, W2, b2)
    assert out.shape == (seq_len, d_model), f"got {out.shape}"
    print("test_feedforward_shape_and_relu passed")


def test_transformer_block_end_to_end():
    d_model, num_heads, d_ff, seq_len, batch = 16, 4, 64, 6, 2
    block = TransformerBlock(d_model, num_heads, d_ff, seed=0)
    x = np.random.default_rng(0).normal(size=(batch, seq_len, d_model))
    out, weights = block.forward(x)
    assert out.shape == x.shape, f"block should preserve shape, got {out.shape} vs {x.shape}"
    assert not np.any(np.isnan(out)), "NaN in transformer block output"
    print("test_transformer_block_end_to_end passed")




if __name__ == "__main__":
    test_attention_shape()
    test_weights_sum_to_one()
    test_masking_blocks_diagonal()
    test_scaling_reduces_peakiness()
    test_multihead_shape_preserved()
    test_split_combine_heads_are_inverses()
    test_num_heads_must_divide_d_model()
    test_positional_encoding_shape()
    test_positional_encoding_bounded()
    test_positional_encoding_position_zero_is_predictable()
    test_layer_norm_output_stats()
    test_layer_norm_handles_zero_variance()
    test_feedforward_shape_and_relu()
    test_transformer_block_end_to_end()
    print("\nAll Stage 1 tests passed.")
