import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from typing import Any

import chex
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom

from src.constants import *


class TokenEmbedders(nnx.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        rngs: nnx.Rngs,
        shared_decoding: bool = False,
        decode: bool = False,
        dtype=None,
    ):
        self.decode = decode
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.shared_decoding = shared_decoding

        self.token_emb = nnx.Embed(
            num_tokens,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        if not shared_decoding:
            self.token_unemb = nnx.Param(
                jrandom.uniform(rngs.params(), (embed_dim, num_tokens))
            )

    def embed(
        self,
        batch: Any,
        **kwargs,
    ):
        inputs = batch["sequence"]
        input_tokens = self.token_emb(
            inputs,
        )
        output_sequence = input_tokens

        return output_sequence

    def unembed(
        self,
        output_seq: chex.Array,
        **kwargs
    ):
        if self.shared_decoding:
            outputs = output_seq @ lax.stop_gradient(self.token_emb.embedding.T)
        else:
            outputs = output_seq @ self.token_unemb

        return outputs
