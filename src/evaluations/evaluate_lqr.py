import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple
from typing_extensions import Protocol, runtime_checkable

import chex
import dill
import jax
import jax.numpy as jnp
import json
import numpy as np

from flax import nnx
from gymnax.environments import environment, spaces

from src.envs.lqr import (
    EnvParams,
    DiscreteTimeLQR,
    is_controllable,
    is_stable,
)

import src.evaluations.decoding as decoding


dtype = jnp.bfloat16
X_THRES = 1e-2
SIGMA_W = 0.0
STD_X = 1.0
MAX_STEPS_IN_EPISODE = 200
OBS_DIM = 3
ACT_DIM = 3
Dtype = Any
Shape = tuple[int, ...]

@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


class EvalInfo(NamedTuple):
    episode_lengths: chex.Array
    episode_returns: chex.Array
    act_means: chex.Array


class StepState(NamedTuple):
    cache: Any
    rng: chex.PRNGKey
    env_params: EnvParams
    env_state: Any
    last_obs: chex.Array
    act_means: chex.Array
    ep_done: bool = False
    ep_return: float = 0.0
    ep_length: int = 0


class EvalState(NamedTuple):
    cache: Any
    rng: chex.PRNGKey
    env_params: EnvParams
    eval_info: EvalInfo


class EvalConfig(NamedTuple):
    eval_episodes: int
    max_steps_in_episode: int
    deterministic_action: bool
    use_autoregressive: bool
    max_decode_len: int


# DEFAULT
def sample_env_params(key, dim_u, dim_x, num_seeds):
    env_params = {
        "A": [],
        "B": [],
    }
    for seed_i in range(num_seeds):
        key = jax.random.fold_in(key, seed_i)
        env_params["A"].append(jnp.zeros((dim_x, dim_x)))
        env_params["B"].append(jnp.zeros((dim_u, dim_u)))
        
        env_key = key
        while not is_controllable(
            env_params["A"][-1],
            env_params["B"][-1],
        ) or not is_stable(
            env_params["A"][-1],
            env_params["B"][-1],
        ) or np.any(np.sum(env_params["A"][-1], axis=-1) > 1)  or np.any(np.sum(env_params["B"][-1], axis=-1) > 1):
            env_key, _ = jax.random.split(env_key)
            env_params["A"][-1] = jnp.tanh(jax.random.normal(
                jax.random.fold_in(env_key, 0),
                shape=(dim_x, dim_x),
            ))
            env_params["B"][-1] = jnp.tanh(jax.random.normal(
                jax.random.fold_in(env_key, 1),
                shape=(dim_u, dim_u),
            ))
    return {
        **{k: jnp.stack(v) for k, v in env_params.items()},
        "Q": jnp.concatenate([jnp.diag(
            1.0 - jax.random.uniform(
                jax.random.fold_in(jax.random.fold_in(key, 2), seed),
                shape=(dim_x),
            )
        )[None] for seed in range(num_seeds)], axis=0),
        "R": jnp.concatenate([jnp.diag(
            1.0 - jax.random.uniform(
                jax.random.fold_in(jax.random.fold_in(key, 3), seed),
                shape=(dim_u),
            )
        )[None] for seed in range(num_seeds)], axis=0),
    }


def make_model_funcs(
    model: nnx.Module,
    observation_space: spaces.Space,
    eval_config: EvalConfig,
):
    if eval_config.use_autoregressive:
        graphdef, _, rest = nnx.split(model, nnx.Cache, ...)
        def decode(batch, cache):
            module = nnx.merge(graphdef, cache, rest)
            module.set_attributes(deterministic=True, decode=True)
            out = module(batch)
            cache = nnx.state(module, nnx.Cache)
            return out, cache
        
        def init_cache():
            if model.use_sink_token:
                _, cache = decode({"sink": 1}, nnx.state(model, nnx.Cache))
            else:
                cache = nnx.state(model, nnx.Cache)
            return cache

        return decode, init_cache

    return decoding.make_decode_funcs(
        model,
        eval_config.max_decode_len,
        observation_space.shape,
        [ACT_DIM,],
        dtype,
    )


def evaluate_single_env(
    rng: chex.PRNGKey,
    model: nnx.Module,
    env: environment.Environment,
    env_params: EnvParams,
    eval_config: EvalConfig,
):
    env_params = EnvParams(
        x_thres=X_THRES,
        max_steps_in_episode=MAX_STEPS_IN_EPISODE,
        sigma_w=SIGMA_W,
        std_x=STD_X,
        A=env_params["A"],
        B=env_params["B"],
        Q=env_params["Q"],
        R=env_params["R"],
    )
    decode, init_cache = make_model_funcs(
        model,
        env.observation_space(env_params),
        eval_config,
    )

    def rollout(ep_i: int, eval_state: EvalState):
        def step(step_state: StepState):
            rng, rng_step = jax.random.split(step_state.rng, 2)

            act_mean, cache = decode({"state": step_state.last_obs[None, None],}, step_state.cache)
            act_mean = act_mean[:, -1]

            action = jax.lax.cond(
                eval_config.deterministic_action,
                lambda rng_step, act_mean: act_mean,
                lambda rng_step, act_mean: (act_mean + jax.random.normal(rng_step, shape=act_mean.shape) * 1e-5).astype(dtype),
                rng_step,
                act_mean,
            )[0]

            obs, env_state, reward, done, _ = env.step(
                rng_step, step_state.env_state, action, step_state.env_params
            )
            _, cache = decode({"action": action[None, None],}, cache)
            _, cache = decode({"reward": reward[None, None],}, cache)
            
            step_state = StepState(
                cache=cache,
                rng=rng,
                env_params=env_params,
                env_state=env_state,
                last_obs=obs,
                act_means=step_state.act_means.at[step_state.ep_length].set(act_mean[0]),
                ep_done=done,
                ep_return=step_state.ep_return + reward.squeeze(),
                ep_length=step_state.ep_length + 1,
            )
            return step_state

        rng = jax.random.fold_in(eval_state.rng, ep_i)
        rng_reset, rng_step = jax.random.split(rng)

        obs, env_state = env.reset(rng_reset, env_params)

        step_state = StepState(
            cache=eval_state.cache,
            rng=rng_step,
            env_params=env_params,
            env_state=env_state,
            last_obs=obs,
            act_means=jnp.zeros((eval_config.max_steps_in_episode, ACT_DIM)),
        )
        step_state = jax.lax.while_loop(
            lambda s: jnp.logical_and(
                s.ep_length < eval_config.max_steps_in_episode,
                jnp.logical_not(s.ep_done),
            ),
            step,
            step_state,
        )

        eval_state = EvalState(
            step_state.cache,
            rng=rng,
            env_params=eval_state.env_params,
            eval_info=EvalInfo(
                episode_lengths=eval_state.eval_info.episode_lengths.at[ep_i].set(step_state.ep_length),
                episode_returns=eval_state.eval_info.episode_returns.at[ep_i].set(step_state.ep_return),
                act_means=eval_state.eval_info.act_means.at[ep_i].set(step_state.act_means),
            )
        )

        return eval_state

    cache = init_cache()

    eval_state = EvalState(
        cache=cache,
        rng=rng,
        env_params=env_params,
        eval_info=EvalInfo(
            episode_lengths=jnp.zeros(eval_episodes,),
            episode_returns=jnp.zeros(eval_episodes,),
            act_means=jnp.zeros((
                eval_episodes,
                eval_config.max_steps_in_episode,
                ACT_DIM,
            )),
        )
    )

    eval_state = jax.lax.fori_loop(
        0,
        eval_episodes,
        rollout,
        eval_state,
    )

    return eval_state


@partial(jax.jit, static_argnames=("model", "env", "eval_config"))
def evaluate(
    rng: chex.PRNGKey,
    model: nnx.Module,
    env: environment.Environment,
    env_params: EnvParams,
    eval_config: EvalConfig,
) -> Tuple[chex.Array, chex.Array]:
    """
    Evaluate a policy given by `model` on `eval_episodes` episodes.
    """

    rngs = jax.random.split(rng, num_envs)
    vmap_evaluate_single_env = jax.vmap(
        evaluate_single_env,
        in_axes=(0, None, None, 0, None),
    )
    return vmap_evaluate_single_env(rngs, model, env, env_params, eval_config)


def main(
    max_decode_len: int,
    learner_path: str,
    eval_seed: int,
    num_envs: int,
    eval_episodes: int,
    deterministic_action: bool,
    use_autoregressive: bool,
    half_precision: bool = True,
):
    config_dict = json.load(open(os.path.join(learner_path, "config.json"), "r"))
    embed_dim = config_dict["model_config"]["model_kwargs"]["embed_dim"]

    last_step = sorted(os.listdir(os.path.join(learner_path, "models")))[-1]
    train_state = dill.load(
        open(os.path.join(learner_path, "models", last_step), "rb")
    )
    model = nnx.merge(
        train_state.graphdef,
        train_state.params,
        train_state.rest,
    )

    rng = jax.random.PRNGKey(eval_seed)
    rng, _ = jax.random.split(rng)

    env = DiscreteTimeLQR(dim_x=OBS_DIM, dim_u=ACT_DIM)

    model.eval()
    model.set_attributes(deterministic=True, decode=use_autoregressive)

    if use_autoregressive:
        for _path, m in model.iter_modules():
            if isinstance(m, HasCache):
                input_shape = (
                    1,
                    int(model.use_sink_token) + max_decode_len * 3,
                    embed_dim,
                )
                m.init_cache(input_shape, dtype=dtype)

    eval_config = EvalConfig(
        eval_episodes=eval_episodes,
        max_steps_in_episode=MAX_STEPS_IN_EPISODE,
        deterministic_action=deterministic_action,
        use_autoregressive=use_autoregressive,
        max_decode_len=max_decode_len,
    )

    env_params = sample_env_params(rng, OBS_DIM, ACT_DIM, num_envs)
    eval_state = evaluate(
        rng,
        model,
        env,
        env_params,
        eval_config,
    )

    eval_info = eval_state.eval_info
    dill.dump(
        {
            **{k: np.array(v) for k, v in eval_info._asdict().items()},
            "eval_config": {
                k: v for k, v in eval_config._asdict().items()
            },
            "env_params": env_params,
        },
        open(os.path.join(learner_path, "eval_info.dill"), "wb"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner_path", type=str, required=True)
    args = parser.parse_args()

    eval_seed = 40
    num_envs = 5
    eval_episodes = 500
    max_decode_len = 500
    deterministic_action = False
    use_autoregressive = False

    if use_autoregressive:
        max_decode_len = eval_episodes * MAX_STEPS_IN_EPISODE

    # learner_path = os.path.join(base_path, algo_name, run_name)
    learner_path = args.learner_path

    main(
        max_decode_len,
        learner_path,
        eval_seed,
        num_envs,
        eval_episodes,
        deterministic_action,
        use_autoregressive,
    )
