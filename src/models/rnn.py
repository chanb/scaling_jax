import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from flax.typing import (
    Dtype,
    Initializer,
)
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np

from src.constants import *


class DropoutGRUCell(nnx.RNNCellBase):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        dropout_p: int = 0.0,
        gate_fn: Callable[..., Any] = nnx.sigmoid,
        activation_fn: Callable[..., Any] = nnx.tanh,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        recurrent_kernel_init: Initializer = nnx.initializers.orthogonal(),
        bias_init: Initializer = nnx.initializers.zeros_init(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = nnx.initializers.zeros_init(),
        rngs: nnx.rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.rngs = rngs

        # Combine input transformations into a single linear layer
        self.dense_i = nnx.Linear(
            in_features=in_features,
            out_features=3 * hidden_features,  # r, z, n
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.dense_h = nnx.Linear(
            in_features=hidden_features,
            out_features=3 * hidden_features,  # r, z, n
            use_bias=False,
            kernel_init=self.recurrent_kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, carry: jax.Array, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = carry

        # Compute combined transformations for inputs and hidden state
        x_transformed = self.dropout(self.dense_i(inputs))
        h_transformed = self.dense_h(h)

        # Split the combined transformations into individual components
        xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
        hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)

        # Compute gates
        r = self.gate_fn(xi_r + hh_r)
        z = self.gate_fn(xi_z + hh_z)

        # Compute n with an additional linear transformation on h
        n = self.activation_fn(xi_n + r * hh_n)

        # Update hidden state
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    def initialize_carry(
        self, input_shape: tuple[int, ...], rngs: nnx.rnglib.Rngs | None = None
    ) -> jax.Array:  # type: ignore[override]
        batch_dims = input_shape[:-1]
        if rngs is None:
            rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        return h

    @property
    def num_feature_axes(self) -> int:
        return 1


class InContextGRU(nnx.Module):
    """A GRU for in-context learning."""

    def __init__(
        self,
        embed_dim: int,
        embedder_cls: Callable,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
        use_sink_token: bool = True,
        **kwargs,
    ) -> None:
        self.decode = decode
        self.use_sink_token = use_sink_token
        self.embedders = embedder_cls()

        if use_sink_token:
            self.sink_token = nnx.Embed(
                1,
                embed_dim,
                rngs=rngs,
                dtype=dtype,
            )
        self.gru = nnx.RNN(
            DropoutGRUCell(
                in_features=embed_dim,
                hidden_features=embed_dim,
                rngs=rngs,
                dtype=dtype,
            )
        )
        
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
        token_seq = self.gru(token_seq)
        outputs = self.embedders.unembed(
            token_seq[:, int(self.use_sink_token):]
        )

        return outputs
