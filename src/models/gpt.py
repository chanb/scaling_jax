import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from typing import Callable, Any

import jax.numpy as jnp
import numpy as np

from src.constants import *


class GPTBlock(nnx.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        widening_factor,
        *,
        rngs,
        use_causal_mask=True,
        dtype=None,
    ):
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

    def __call__(self, x):
        mask = nnx.make_causal_mask(x[..., 0]) * self.use_causal_mask
        mask = mask + jnp.ones_like(mask) * (1 - self.use_causal_mask)
        normed_x = self.ln_1(x)
        attention_out = self.attention(normed_x, normed_x, normed_x, mask=mask)
        x = x + attention_out
        normed_x = nnx.gelu(self.dense_1(self.ln_2(x)))
        x = x + self.dense_2(normed_x)

        self.sow(nnx.Intermediate, "attention_out", attention_out)
        self.sow(nnx.Intermediate, "block_out", x)

        return x


class GPT(nnx.Module):
    def __init__(
        self,
        num_blocks,
        num_heads,
        embed_dim,
        widening_factor,
        *,
        rngs,
        use_causal_mask=True,
        dtype=None,
    ):
        layers = []
        for _ in range(num_blocks):
            layers.append(
                GPTBlock(
                    num_heads,
                    embed_dim,
                    widening_factor,
                    rngs=rngs,
                    use_causal_mask=use_causal_mask,
                    dtype=dtype,
                )
            )
        self.gpt = nnx.Sequential(*layers)
        self.ln = nnx.LayerNorm(embed_dim, rngs=rngs, dtype=dtype,)

    def __call__(self, x):
        x = self.gpt(x)
        x = self.ln(x)
        return x


class InContextGPT(nnx.Module):
    """A GPT for in-context learning."""

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        widening_factor: int,
        encoder_cls: Callable,
        predictor_cls: Callable,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
        use_sink_token: bool = True,
        **kwargs,
    ) -> None:
        self.decode = decode
        self.use_sink_token = use_sink_token
        self.encoder = encoder_cls()

        if use_sink_token:
            self.sink_token = nnx.Embed(
                1,
                embed_dim,
                rngs=rngs,
                dtype=dtype,
            )
        self.gpt = GPT(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            widening_factor=widening_factor,
            rngs=rngs,
            use_causal_mask=True,
            dtype=dtype,
        )

        self.predictor = predictor_cls()
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def __call__(
        self,
        batch: Any,
    ):
        if self.decode:
            if "sink" not in batch or not self.use_sink_token:
                token_seq = self.encoder(batch)
            else:
                token_seq = self.sink_token(
                    np.zeros((batch["sink"], 1), dtype=int),
                )
        else:
            token_seq = self.encoder(batch)
            if self.use_sink_token:
                sink_token = self.sink_token(
                    np.zeros((len(token_seq), 1), dtype=int),
                )
                token_seq = jnp.concatenate(
                    (sink_token, token_seq),
                    axis=1,
                )
        token_seq = self.gpt(token_seq)
        outputs = self.predictor(token_seq[:, int(self.use_sink_token):])

        return outputs
