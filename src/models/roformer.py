import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from flax.typing import Dtype, Shape
from typing import Callable, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.constants import *


"Referenced from https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/roformer/modeling_roformer.py"
def apply_rotary_position_embeddings(
    sinusoidal_pos_embed: jax.Array,
    queries: jax.Array,
    keys: jax.Array,
    num_heads: int,
    embed_dim_per_head: int,
):
    # MultiHeadAttention treats q, k, v as (batch, seq_len, num_heads, head_dim)
    sin, cos = jnp.split(sinusoidal_pos_embed, 2, axis=-1)
    bs, seq_len, _ = queries.shape

    sin = sin.reshape(
        (bs, seq_len, 1, embed_dim_per_head // 2)
    )
    cos = cos.reshape(
        (bs, seq_len, 1, embed_dim_per_head // 2)
    )

    sin_pos = jnp.stack([sin, sin], axis=-1).reshape(
        (bs, seq_len, 1, embed_dim_per_head)
    )
    cos_pos = jnp.stack([cos, cos], axis=-1).reshape(
        (bs, seq_len, 1, embed_dim_per_head)
    )

    queries = queries.reshape(
        (bs, seq_len, num_heads, embed_dim_per_head)
    )
    keys = keys.reshape(
        (bs, seq_len, num_heads, embed_dim_per_head)
    )

    rotate_half_queries = jnp.stack(
        (-queries[..., 1::2], queries[..., ::2]),
        axis=-1,
    ).reshape(queries.shape)
    queries = queries * cos_pos + rotate_half_queries * sin_pos

    rotate_half_keys = jnp.stack(
        (-keys[..., 1::2], keys[..., ::2]),
        axis=-1,
    ).reshape(keys.shape)
    keys = keys * cos_pos + rotate_half_keys * sin_pos

    queries = queries.reshape((bs, seq_len, -1))
    keys = keys.reshape((bs, seq_len, -1))

    return queries, keys


def get_sinusoidal_position_embedding(
    positions: jax.Array,
    embed_dim_per_head: int,
    n_value: float = 10000.0,
) -> jax.Array:
    # positions: [batch, sequence_length,]
    thetas = positions.flatten()[:, None] / jnp.power(
        n_value, 2 * (jnp.arange(embed_dim_per_head) // 2) / embed_dim_per_head
    )[None]
    thetas = thetas.reshape((*positions.shape, embed_dim_per_head))

    out_sin = jnp.sin(thetas[..., 0::2])
    out_cos = jnp.cos(thetas[..., 1::2])
    return jnp.concatenate((out_sin, out_cos), axis=-1)


class RoformerBlock(nnx.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        widening_factor,
        *,
        rngs,
        use_causal_mask=True,
        decode: bool = False,
        dtype=None,
        n_value=10000.0,
    ):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        self.n_value = n_value
        self.decode = decode
        self.use_causal_mask = use_causal_mask
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            embed_dim,
            decode=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_normal(),
                ("tensor", "fsdp")
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=1.0),
                ("fsdp",)
            ),
        )
        self.ln_1 = nnx.LayerNorm(
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        self.ln_2 = nnx.LayerNorm(
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )
        self.dense_1 = nnx.Linear(
            embed_dim,
            embed_dim * widening_factor,
            rngs=rngs,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_normal(),
                ("tensor", "fsdp")
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=1.0),
                ("fsdp",)
            ),
        )
        self.dense_2 = nnx.Linear(
            embed_dim * widening_factor,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_normal(),
                ("tensor", "fsdp")
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=1.0),
                ("fsdp",)
            ),
        )
        self.cache_pos: nnx.Cache[jax.Array] | None = None

    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
        self.cache_pos = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    def __call__(self, x):
        mask = nnx.make_causal_mask(x[..., 0]) * self.use_causal_mask
        mask = mask + jnp.ones_like(mask) * (1 - self.use_causal_mask)
        normed_x = self.ln_1(x)

        # Apply rotary embedding
        if self.decode:
            positions = jnp.full(
                x.shape[:-1],
                fill_value=self.cache_pos[...],
            )
            self.cache_pos.value += 1
        else:
            positions = jnp.tile(
                jnp.arange(normed_x.shape[1]),
                reps=(normed_x.shape[0], 1),
            )
        sinusoidal_pos_embed = get_sinusoidal_position_embedding(
            positions,
            self.embed_dim_per_head,
            self.n_value,
        )
        q, k = apply_rotary_position_embeddings(
            sinusoidal_pos_embed=sinusoidal_pos_embed,
            queries=normed_x,
            keys=normed_x,
            num_heads=self.num_heads,
            embed_dim_per_head=self.embed_dim_per_head,
        )
        v = normed_x

        attention_out = self.attention(q, k, v, mask=mask)
        x = x + attention_out
        normed_x = nnx.gelu(self.dense_1(self.ln_2(x)))
        x = x + self.dense_2(normed_x)

        self.sow(nnx.Intermediate, "attention_out", attention_out)
        self.sow(nnx.Intermediate, "block_out", x)

        return x


class Roformer(nnx.Module):
    def __init__(
        self,
        num_blocks,
        num_heads,
        embed_dim,
        widening_factor,
        *,
        rngs,
        use_causal_mask=True,
        decode: bool = False,
        dtype=None,
    ):
        layers = []
        for _ in range(num_blocks):
            layers.append(
                RoformerBlock(
                    num_heads,
                    embed_dim,
                    widening_factor,
                    rngs=rngs,
                    use_causal_mask=use_causal_mask,
                    decode=decode,
                    dtype=dtype,
                )
            )
        self.roformer = nnx.Sequential(*layers)
        self.ln = nnx.LayerNorm(embed_dim, rngs=rngs, dtype=dtype,)

    def __call__(self, x):
        x = self.roformer(x)
        x = self.ln(x)
        return x


class InContextRoformer(nnx.Module):
    """A Roformer for in-context learning."""

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        widening_factor: int,
        embedder_cls: Callable,
        pos_enc_cls: Callable,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype = None,
        use_sink_token: bool = True,
        **kwargs,
    ) -> None:
        self.decode = decode
        self.use_sink_token = use_sink_token
        self.embedders = embedder_cls()
        self.pos_enc = pos_enc_cls()

        if use_sink_token:
            self.sink_token = nnx.Embed(
                1,
                embed_dim,
                rngs=rngs,
                dtype=dtype,
            )

        self.roformer = Roformer(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            widening_factor=widening_factor,
            rngs=rngs,
            use_causal_mask=True,
            decode=decode,
            dtype=dtype,
        )
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def __call__(
        self,
        batch: Any,
    ):
        if self.decode:
            if "sink" not in batch or not self.use_sink_token:
                token_seq = self.embedders.embed(batch)
            else:
                token_seq = self.sink_token(
                    np.zeros((batch["sink"], 1), dtype=int),
                )
        else:
            token_seq = self.embedders.embed(batch)
            if self.use_sink_token:
                sink_token = self.sink_token(
                    np.zeros((len(token_seq), 1), dtype=int),
                )
                token_seq = jnp.concatenate(
                    (sink_token, token_seq),
                    axis=1,
                )
        token_seq = self.pos_enc(token_seq)
        token_seq = self.roformer(token_seq)
        outputs = self.embedders.unembed(
            token_seq[:, int(self.use_sink_token):]
        )

        return outputs
