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
from src.models.common import MLP, identity


class RegressionEmbedders(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        rngs: nnx.Rngs,
        shared_decoding: bool = False,
        decode: bool = False,
        dtype=None,
    ):
        self.decode = decode
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.shared_decoding = shared_decoding

        self.input_emb = nnx.Linear(
            input_dim,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        self.output_emb = nnx.Param(
            jrandom.uniform(rngs.params(), (1, embed_dim))
        )

        if not shared_decoding:
            self.output_unemb = nnx.Param(
                jrandom.uniform(rngs.params(), (embed_dim, 1))
            )

    def embed(
        self,
        batch: Any,
        **kwargs,
    ):
        if "example" in batch:
            inputs = batch["example"]
            input_tokens = self.input_emb(
                inputs,
            )
            output_sequence = input_tokens
        
        if "target" in batch:
            targets = batch["target"]
            target_tokens = targets @ self.output_emb
            output_sequence = target_tokens

        if self.decode:
            return output_sequence
        else:
            (batch_size, seq_len) = inputs.shape[:2]
            output_sequence = jnp.concatenate((
                input_tokens[:, :, None, :],
                target_tokens[:, :, None, :],
            ), axis=2).reshape((
                batch_size,
                2 * seq_len,
                self.embed_dim, # D
            ))

        return output_sequence

    def unembed(
        self,
        output_seq: chex.Array,
        **kwargs
    ):
        if self.shared_decoding:
            outputs = output_seq @ lax.stop_gradient(self.output_emb.T)
        else:
            outputs = output_seq @ self.output_unemb

        if self.decode:
            return outputs
        else:
            return outputs[:, ::2]
