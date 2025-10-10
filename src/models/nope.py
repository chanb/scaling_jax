from flax import nnx

import jax


class NoPE(nnx.Module):
    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        return x
