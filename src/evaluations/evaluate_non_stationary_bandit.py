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
from gymnax.environments import environment

from src.envs.bernoulli_bandit import BernoulliBandit, EnvParams, EnvState


Dtype = Any
Shape = tuple[int, ...]

@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


class EvalInfo(NamedTuple):
    episode_lengths: chex.Array
    episode_returns: chex.Array
    action_counts: chex.Array
    logits: chex.Array


class StepState(NamedTuple):
    cache: Any
    rng: chex.PRNGKey
    env_params: EnvParams
    env_state: EnvState
    last_obs: chex.Array
    logits: chex.Array
    act_counts: chex.Array
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
    switch_freq: int
    max_steps_in_episode: int
    num_arms: int
    num_envs: int
    num_switches: int
    sample_env_params: Callable
    deterministic_action: bool
    use_autoregressive: bool
    max_decode_len: int


def sample_env_params(key, num_arms):
    reward_probs = jax.random.beta(
        key,
        a=0.2,
        b=0.2,
        shape=(
            num_arms,
        ),
    )

    return EnvParams(reward_probs=reward_probs)


def evaluate_single_env(
    rng: chex.PRNGKey,
    model: nnx.Module,
    env: environment.Environment,
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
    else:
        def decode(batch, cache):
            new_cache = {
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
            }
            out = model(new_cache)
            return out, new_cache

        def init_cache():
            cache = {
                "state": jnp.zeros(
                    (
                        1,
                        eval_config.max_decode_len,
                        *env.observation_space(
                            jnp.zeros((eval_config.num_arms,))
                        ).shape,
                    )
                ),
                "action": jnp.zeros(
                    (1, eval_config.max_decode_len,),
                    dtype=int,
                ),
                "reward": jnp.zeros(
                    (1, eval_config.max_decode_len,)
                ),
            }
            return cache

    def rollout(ep_i: int, eval_state: EvalState):
        def step(step_state: StepState):
            rng, rng_step = jax.random.split(step_state.rng, 2)

            logits, cache = decode({"state": step_state.last_obs[None, None, None],}, step_state.cache)
            logits = logits[:, -1]

            action = jax.lax.cond(
                eval_config.deterministic_action,
                lambda rng_step, logits: jnp.argmax(logits, axis=-1),
                jax.random.categorical,
                rng_step,
                logits,
            )

            obs, env_state, reward, done, _ = env.step(
                rng_step, step_state.env_state, action, step_state.env_params
            )
            _, cache = decode({"action": action[:, None],}, cache)
            _, cache = decode({"reward": reward[:, None],}, cache)
            
            step_state = StepState(
                cache=cache,
                rng=rng,
                env_params=env_params,
                env_state=env_state,
                last_obs=obs,
                logits=step_state.logits.at[step_state.ep_length].set(logits[0]),
                ep_done=done,
                ep_return=step_state.ep_return + reward.squeeze(),
                ep_length=step_state.ep_length + 1,
                act_counts=step_state.act_counts.at[action].set(step_state.act_counts[action] + 1),
            )
            return step_state

        rng = jax.random.fold_in(eval_state.rng, ep_i)
        rng_reset, rng_step = jax.random.split(rng)

        # Sample new bandit instance
        curr_instance = ep_i // eval_config.switch_freq
        env_params = jax.lax.cond(
            ep_i % eval_config.switch_freq == 0,
            eval_config.sample_env_params,
            lambda x: EnvParams(reward_probs=eval_state.env_params[curr_instance]),
            rng,
        )

        obs, env_state = env.reset(rng_reset, env_params)

        step_state = StepState(
            cache=eval_state.cache,
            rng=rng_step,
            env_params=env_params,
            env_state=env_state,
            last_obs=obs,
            logits=jnp.zeros((eval_config.max_steps_in_episode, eval_config.num_arms)),
            act_counts=jnp.zeros(eval_config.num_arms),
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
            env_params=eval_state.env_params.at[curr_instance].set(env_params.reward_probs),
            eval_info=EvalInfo(
                episode_lengths=eval_state.eval_info.episode_lengths.at[ep_i].set(step_state.ep_length),
                episode_returns=eval_state.eval_info.episode_returns.at[ep_i].set(step_state.ep_return),
                action_counts=eval_state.eval_info.action_counts + step_state.act_counts,
                logits=eval_state.eval_info.logits.at[ep_i].set(step_state.logits),
            )
        )

        return eval_state

    cache = init_cache()

    eval_state = EvalState(
        cache=cache,
        rng=rng,
        env_params=jnp.zeros((eval_config.num_switches, eval_config.num_arms)),
        eval_info=EvalInfo(
            episode_lengths=jnp.zeros(eval_episodes,),
            episode_returns=jnp.zeros(eval_episodes,),
            action_counts=jnp.zeros(eval_config.num_arms),
            logits=jnp.zeros((
                eval_episodes,
                eval_config.max_steps_in_episode,
                eval_config.num_arms,
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
    eval_config: EvalConfig,
) -> Tuple[chex.Array, chex.Array]:
    """
    Evaluate a policy given by `model` on `eval_episodes` episodes.
    """

    rngs = jax.random.split(rng, num_envs)
    vmap_evaluate_single_env = jax.vmap(
        evaluate_single_env,
        in_axes=(0, None, None, None),
    )
    return vmap_evaluate_single_env(rngs, model, env, eval_config)


def main(
    max_decode_len: int,
    learner_path: str,
    eval_seed: int,
    num_envs: int,
    eval_episodes: int,
    switch_freq: int,
    deterministic_action: bool,
    use_autoregressive: bool,
    half_precision: bool = True
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

    env = BernoulliBandit()

    dtype = jnp.bfloat16 if half_precision else jnp.float32

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
        switch_freq=switch_freq,
        max_steps_in_episode=1,
        num_envs=num_envs,
        num_arms=config_dict["model_config"]["model_kwargs"]["output_dim"],
        num_switches=int(np.ceil(eval_episodes / switch_freq)),
        sample_env_params=partial(
            sample_env_params,
            num_arms=config_dict["model_config"]["model_kwargs"]["output_dim"],
        ),
        deterministic_action=deterministic_action,
        use_autoregressive=use_autoregressive,
        max_decode_len=max_decode_len * 3,
    )

    eval_state = evaluate(
        rng,
        model,
        env,
        eval_config,
    )

    eval_info = eval_state.eval_info
    dill.dump(
        {
            **{k: np.array(v) for k, v in eval_info._asdict().items()},
            "env_params": np.array(eval_state.env_params),
            "switch_freq": eval_config.switch_freq,
            "num_envs": eval_config.num_envs,
            "num_arms": eval_config.num_arms,
        },
        open(os.path.join(learner_path, "eval_info.dill"), "wb"),
    )


if __name__ == "__main__":
    base_path = "/home/bryanpu1/projects/aaai_2026/scaling_jax/results"
    algo_name = "bandit_ad"
    run_name = "adamw-06-06-25_09_56_26-8373a959-1e98-4fe9-bedb-bd9a3e42a6b5"

    eval_seed = 40
    num_envs = 20
    eval_episodes = 500
    switch_freq = 100
    max_decode_len = 100
    deterministic_action = False
    use_autoregressive = False

    if use_autoregressive:
        max_decode_len = eval_episodes

    learner_path = os.path.join(base_path, algo_name, run_name)

    main(
        max_decode_len,
        learner_path,
        eval_seed,
        num_envs,
        eval_episodes,
        switch_freq,
        deterministic_action,
        use_autoregressive,
    )
