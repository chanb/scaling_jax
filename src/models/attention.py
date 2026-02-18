import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx.module import Module
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import (
    Dtype,
    PrecisionLike,
)
from typing import Callable

Array = jax.Array


def _quiet_softmax(
    x: Array,
    axis: int | tuple[int, ...] | None = -1,
    where: Array | None = None,
    initial: Array | None = -jnp.inf
) -> Array:
    x_max = jax.lax.stop_gradient(
        jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    )
    x_safe = x if where is None else jnp.where(where, x, initial)
    unnormalized = jnp.exp(x_safe - x_max)
    result = unnormalized / (
        jnp.exp(-x_max)
        + jnp.sum(unnormalized, axis, where=where, keepdims=True)
    )
    if where is not None:
        result = jnp.where(where, result, 0)
    return result

def general_dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    attention_fn: Callable | None = None,
):
    """
    Computes general dot-product attention weights given query and key.
    """
    query, key = promote_dtype((query, key), dtype=dtype)  # type: ignore[bad-unpacking]
    dtype = query.dtype

    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
        '...qhd,...khd->...hqk', query, key, precision=precision
    )

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = 0.0 if attention_fn is None else jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    if attention_fn is not None:
        attn_weights = attention_fn(attn_weights).astype(dtype)

    if module:
        module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            attn_weights = attn_weights * multiplier

    return attn_weights


def quiet_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
):
    """
    Computes quiet dot-product attention given query, key, and value.
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = general_dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        module,
        _quiet_softmax,
    )

    # return weighted sum over values for each query position
    return jnp.einsum(
        '...hqk,...khd->...qhd', attn_weights, value, precision=precision
    )

def linear_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
):
    """
    Computes linear dot-product attention given query, key, and value.
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = general_dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        module,
    )

    # return weighted sum over values for each query position
    return jnp.einsum(
        '...hqk,...khd->...qhd', attn_weights, value, precision=precision
    )
