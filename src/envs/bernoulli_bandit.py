"""JAX implementation of a Bernoulli bandit environment as in Wang et al. 2017."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    reward_probs: jax.Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    reward_probs: tuple[float] = (0.5, 0.5)


class BernoulliBandit(environment.Environment[EnvState, EnvParams]):
    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Sample bernoulli reward, increase counter, construct input."""
        reward = jax.random.bernoulli(key, state.reward_probs[action]).astype(jnp.int32)
        state = EnvState(
            reward_probs=state.reward_probs,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        state = EnvState(
            reward_probs=jnp.array(params.reward_probs),
            time=0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        return jnp.array(0)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = True
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "BernoulliBandit-custom"

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(params.reward_probs))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0.0, 1.0, (1,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "reward_probs": spaces.Box(0, 1, (len(params.reward_probs),), jnp.float32),
            }
        )
