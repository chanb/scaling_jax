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
        if getattr(model, "use_sink_token", False):
            _, cache = decode({"sink": batch_size}, nnx.state(model, nnx.Cache))
        else:
            cache = nnx.state(model, nnx.Cache)
        return cache

    return decode, init_cache


# def decode(batch, cache):
#     new_cache = {
#         "state": jnp.concatenate(
#             (cache["state"][:, 1:], batch["state"]),
#             axis=1,
#         ) if "state" in batch else cache["state"],
#         "action": jnp.concatenate(
#             (cache["action"][:, 1:], batch["action"]),
#             axis=1,
#         ) if "action" in batch else cache["action"],
#         "reward": jnp.concatenate(
#             (cache["reward"][:, 1:], batch["reward"]),
#             axis=1,
#         ) if "reward" in batch else cache["reward"],
#     }
#     out = model(new_cache)
#     return out, new_cache

# def init_cache():
#     cache = {
#         "state": jnp.zeros(
#             (
#                 1,
#                 eval_config.max_decode_len,
#                 *env.observation_space(
#                     jnp.zeros((eval_config.num_arms,))
#                 ).shape,
#             )
#         ),
#         "action": jnp.zeros(
#             (1, eval_config.max_decode_len,),
#             dtype=int,
#         ),
#         "reward": jnp.zeros(
#             (1, eval_config.max_decode_len,)
#         ),
#     }
#     return cache
