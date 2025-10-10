import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
from flax.nnx.module import first_from
from flax.typing import (
    Dtype,
    Shape,
)

class SinusoidalPE(nnx.Module):
    """
    Default positional encoding used in Transformers. More correct implementation following Chan et al.?
    Reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype: jnp.dtype = jnp.float32,
        period: float = 30.0,
    ):
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dtype = dtype
        self.decode = decode
        self.period = period
        pe = np.zeros((self.max_len, self.embed_dim))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.log(self.period) * (-np.arange(0, self.embed_dim, 2) / self.embed_dim)
        )
        half_dim = self.embed_dim // 2
        pe[:, :half_dim] = np.sin(position * div_term)
        pe[:, half_dim:] = np.cos(position * div_term)
        self.pe = nnx.Variable(pe[None])

        self.cache_pos: nnx.Cache[jax.Array] | None = None

    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
        self.cache_pos = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    def __call__(
        self,
        x: jax.Array,
        *,
        decode: bool | None = None,
        **kwargs,
    ):
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
                as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            if (
                self.cache_pos is None
            ):
                raise ValueError(
                'Autoregressive cache not initialized, call ``init_cache`` first.'
                )
            pos = self.cache_pos[...]
            x = x + self.pe[:, [pos]]
            self.cache_pos.value += 1
        else:
            x = x + self.pe[:, :x.shape[-2]]

        return x
