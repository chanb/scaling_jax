import chex
import jax
import jax.numpy as jnp

from flax import nnx
from gymnax.environments import environment


import jax

def cache_replace_loc(batch, cache):
    return {
        "state": (
            cache["state"].at[:, [cache["count"]]].set(batch["state"])
            if "state" in batch else
            cache["state"]
        ),
        "action": (
            cache["action"].at[:, [cache["count"]]].set(batch["action"])
            if "action" in batch else
            cache["action"]
        ),
        "reward": (
            cache["reward"].at[:, [cache["count"]]].set(batch["reward"])
            if "reward" in batch else
            cache["reward"]
        ),
        "count": cache["count"] + 1 if "reward" in batch else cache["count"],
    }

def cache_concat(batch, cache):
    return {
        "state": jnp.concatenate(
            (cache["state"][:, 1:], batch["state"]),
            axis=1,
        ) if "state" in batch else cache["state"],
        "action": jnp.concatenate(
            (cache["action"][:, 1:], batch["action"]),
            axis=1,
        ) if "action" in batch else cache["action"],
        "reward": jnp.concatenate(
            (cache["reward"][:, 1:], batch["reward"]),
            axis=1,
        ) if "reward" in batch else cache["reward"],
        "count": cache["count"] + 1 if "reward" in batch else cache["count"],
    }

def make_decode_funcs(
    model: nnx.Module,
    max_decode_len: int,
    obs_dim: chex.Array,
    act_dim: chex.Array,
):
    def decode(batch, cache):
        new_cache = jax.lax.cond(
            cache["count"] < max_decode_len,
            cache_replace_loc,
            cache_concat,
            batch,
            cache,
        )
        out = model(new_cache)
        out = jax.lax.cond(
            cache["count"] < max_decode_len,
            lambda: out[:, [3 * cache["count"]]],
            lambda: out[:, [-3]],
        )
        return out, new_cache

    def init_cache():
        cache = {
            "state": jnp.zeros(
                (1, max_decode_len, *obs_dim,)
            ),
            "action": jnp.zeros(
                (1, max_decode_len, *act_dim),
                dtype=int,
            ),
            "reward": jnp.zeros(
                (1, max_decode_len,)
            ),
            "count": 0,
        }
        return cache
    
    return decode, init_cache