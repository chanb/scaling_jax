from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    x: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    x_thres: float=1e-1
    max_steps_in_episode: float=200
    sigma_w: float=0
    std_x: float=1.0
    A: jax.Array=jnp.eye(2)
    B: jax.Array=jnp.eye(2)
    Q: jax.Array=jnp.eye(2)
    R: jax.Array=jnp.eye(2)


class DiscreteTimeLQR(environment.Environment[EnvState, EnvParams]):
    def __init__(self, dim_x, dim_u):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u

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
        """Performs step transitions in the environment."""

        # Compute reward
        reward = -(state.x.T @ params.Q @ state.x + action.T @ params.R @ action)

        # Compute next state: x' = Ax + Bu + sigma_w * w, w ~ N(0, I)
        next_state = params.A @ state.x + params.B @ action
        next_state = next_state + params.sigma_w * jax.random.normal(key, self.dim_x)
        next_t = state.time + 1

        state = EnvState(
            x=next_state,
            time=next_t
        )

        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Performs resetting of environment."""
        init_state = jnp.tanh(jax.random.normal(key, shape=(self.dim_x,))) * params.std_x
        state = EnvState(
            x=init_state,
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return jnp.array(state.x)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done_threshold = jnp.all(jnp.abs(state.x) <= params.x_thres)
        done_diverge = jnp.any(jnp.isnan(state.x))

        done = jnp.logical_or(done_steps, done_threshold)
        done = jnp.logical_or(done, done_diverge)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "DiscreteTimeLQR-v0"

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(-1.0, 1.0, (self.dim_u,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0.0, jnp.inf, (self.dim_x,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "x": spaces.Box(0.0, jnp.inf, (self.dim_x,), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def is_controllable(A, B):
    dim_x = A.shape[0]
    W = np.copy(B)
    for _ in range(1, dim_x):
        W = np.hstack((W, A @ B))
        A = A @ B
    return np.linalg.matrix_rank(W) == dim_x


def is_stable(A, B):
    pos_eig_vals = np.linalg.eigvals(A + B @ np.eye(B.shape[0]))
    neg_eig_vals = np.linalg.eigvals(A + B @ -np.eye(B.shape[0]))
    return (
        np.all(np.abs(pos_eig_vals) < 1)
        and np.all(np.abs(neg_eig_vals) < 1)
    )
