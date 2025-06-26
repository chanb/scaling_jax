import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from typing import Any
from typing_extensions import Protocol, runtime_checkable

import jax.numpy as jnp

from flax import nnx


Dtype = Any
Shape = tuple[int, ...]

@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


def make_autoregressive(
    model: nnx.Module,
    max_decode_len: int,
    batch_size: int,
    embed_dim: int,
    dtype: Dtype,
    eval_mode: bool,
):
    if eval_mode:
        model.eval()
    model.set_attributes(deterministic=True, decode=True)

    for _, m in model.iter_modules():
        if isinstance(m, HasCache):
            input_shape = (
                batch_size,
                int(getattr(model, "use_sink_token", False)) + max_decode_len,
                embed_dim,
            )
            m.init_cache(input_shape, dtype=dtype)

    graphdef, _, rest = nnx.split(model, nnx.Cache, ...)
    def decode(batch, cache):
        module = nnx.merge(graphdef, cache, rest)
        module.set_attributes(deterministic=True, decode=True)
        out = module(batch)
        cache = nnx.state(module, nnx.Cache)
        return out, cache
    
    def init_cache():
        if model.use_sink_token:
            _, cache = decode({"sink": batch_size}, nnx.state(model, nnx.Cache))
        else:
            cache = nnx.state(model, nnx.Cache)
        return cache

    return model, decode, init_cache
