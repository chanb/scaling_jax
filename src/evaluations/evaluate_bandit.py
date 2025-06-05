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

from flax import nnx
from gymnax.environments import environment

from src.envs.bernoulli_bandit import BernoulliBandit, EnvParams


Dtype = Any
Shape = tuple[int, ...]

@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


class EvalState(NamedTuple):
    cache: Any
    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    act_counts: chex.Array
    done: bool = False
    return_: float = 0.0
    length: int = 0


def evaluate_single(
    model: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    env,
    env_params,
    rng,
    eval_episodes,
    max_steps_in_episode,
):
    graphdef, _, rest = nnx.split(model, nnx.Cache, ...)
    def decode(batch, cache):
        module = nnx.merge(graphdef, cache, rest)
        module.set_attributes(deterministic=True, decode=True)
        out = module(batch)
        cache = nnx.state(module, nnx.Cache)
        return out, cache

    def rollout(ep_i, info):
        def step(state):
            rng, rng_step = jax.random.split(state.rng, 2)

            logits, cache = decode({"state": state.last_obs[None, None, None],}, state.cache)
            logits = logits[:, -1]
            action = jnp.argmax(logits, axis=-1)

            obs, env_state, reward, done, _ = env.step(
                rng_step, state.env_state, action, env_params
            )
            _, cache = decode({"action": action[:, None],}, cache)
            _, cache = decode({"reward": reward[:, None],}, cache)
            
            state = EvalState(
                cache=cache,
                rng=rng,
                env_state=env_state,
                last_obs=obs,
                done=done,
                return_=state.return_ + reward.squeeze(),
                length=state.length + 1,
                act_counts=state.act_counts.at[action].set(state.act_counts[action] + 1),
            )
            return state

        rng = jax.random.fold_in(info["rng"], ep_i)
        rng_reset, rng_eval = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        state = EvalState(
            info["cache"],
            rng_eval,
            env_state,
            obs,
            act_counts=info["eval_info"]["act_counts"],
        )
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(
                s.length < max_steps_in_episode, jnp.logical_not(s.done)
            ),
            step,
            state,
        )

        info["cache"] = state.cache
        info["eval_info"]["episode_length"] = info["eval_info"]["episode_length"].at[ep_i].set(state.length)
        info["eval_info"]["episode_return"] = info["eval_info"]["episode_return"].at[ep_i].set(state.return_)
        info["eval_info"]["act_counts"] = state.act_counts
        return info

    _, cache = decode({"sink": 1}, nnx.state(model, nnx.Cache))

    eval_info = {
        "cache": cache,
        "rng": rng,
        "eval_info":{
            "act_counts": jnp.zeros(len(env_params.reward_probs)),
            "episode_length": jnp.zeros(eval_episodes,),
            "episode_return": jnp.zeros(eval_episodes,),
        }
    }

    _, cache, _ = nnx.split(model, nnx.Cache, ...)

    results = jax.lax.fori_loop(
        0,
        eval_episodes,
        rollout,
        eval_info,
    )

    return results["eval_info"]


@partial(jax.jit, static_argnames=("model", "env", "num_envs", "eval_episodes"))
def evaluate(
    model: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    rng: chex.PRNGKey,
    env: environment.Environment,
    env_params: Any,
    num_envs: int,
    eval_episodes: int = 128,
    max_steps_in_episode: Optional[int] = None,
) -> Tuple[chex.Array, chex.Array]:
    """
    Evaluate a policy given by `model` on `eval_episodes` episodes.
    """
    if max_steps_in_episode is None:
        max_steps_in_episode = env_params.max_steps_in_episode

    seeds = jax.random.split(rng, num_envs)
    vmap_evaluate_single = jax.vmap(evaluate_single, in_axes=(None, None, 0, 0, None, None))
    return vmap_evaluate_single(model, env, env_params, seeds, eval_episodes, max_steps_in_episode)


def main(
    max_decode_len: int,
    learner_path: str,
    eval_seed: int,
    num_envs: int,
    eval_episodes: int,
    half_precision: bool = True
):
    config_dict = json.load(open(os.path.join(learner_path, "config.json"), "r"))
    embed_dim = config_dict["model_config"]["model_kwargs"]["embed_dim"]

    last_step = sorted(os.listdir(os.path.join(learner_path, "models")))[-1]
    train_state = dill.load(open(os.path.join(learner_path, "models", last_step), "rb"))
    model = nnx.merge(
        train_state.graphdef,
        train_state.params,
        train_state.rest,
    )

    rng = jax.random.PRNGKey(eval_seed)
    rng, key = jax.random.split(rng)

    env = BernoulliBandit()
    env_params = jax.random.beta(
        key,
        a=0.2,
        b=0.2,
        shape=(
            num_envs,
            config_dict["model_config"]["model_kwargs"]["output_dim"],
        ),
    )

    dtype = jnp.bfloat16 if half_precision else jnp.float32

    model.eval()
    model.set_attributes(deterministic=True, decode=True)
    for _path, m in model.iter_modules():
        if isinstance(m, HasCache):
            input_shape = (1, 1 + max_decode_len * 3, embed_dim)
            m.init_cache(input_shape, dtype=dtype)

    eval_info = evaluate(
        model,
        rng,
        env,
        jax.vmap(
            lambda x: EnvParams(reward_probs=x)
        )(
            env_params
        ),
        num_envs,
        eval_episodes,
        1,
    )
    eval_info["env_params"] = env_params
    dill.dump(
        eval_info,
        open(os.path.join(learner_path, "eval_info.dill"), "wb"),
    )


if __name__ == "__main__":
    base_path = "/home/bryanpu1/projects/aaai_2026/scaling_jax/results"
    algo_name = "bandit_ad"
    run_name = "debug-06-05-25_09_29_45-4d86f527-fcc1-43b7-aeed-3901b032d33b"

    eval_seed = 42
    num_envs = 2
    eval_episodes = 50
    max_decode_len = 512

    learner_path = os.path.join(base_path, algo_name, run_name)

    main(max_decode_len, learner_path, eval_seed, num_envs, eval_episodes)
