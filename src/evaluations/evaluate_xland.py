import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import dill
import itertools
import jax
import jax.numpy as jnp
import jax.random as jrandom
import json
import numpy as np
import xminigrid

from collections import defaultdict
from flax import nnx
from typing import Any
from typing_extensions import Protocol, runtime_checkable
from tqdm import tqdm

from xminigrid.core.constants import NUM_ACTIONS
from xminigrid.environment import EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from src.datasets.ad_dataset import XMiniGridADDataset

Dtype = Any
Shape = tuple[int, ...]

@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


def evaluate(
    step_fn,
    reset_fn,
    env_params: EnvParams,
    rng_key: jax.Array,
    reset_rng: jax.Array,
    model,
    max_decode_len: int,
    embed_dim: int,
    ruleset_ids: np.ndarray,
    eval_episodes: int,
    half_precision: bool = True,
):
    num_envs = len(ruleset_ids)
    num_episodes = np.zeros(num_envs)
    returns = np.zeros(num_envs)

    eval_info = defaultdict(list)

    timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
    dtype = jnp.bfloat16 if half_precision else jnp.float32

    model.eval()
    model.set_attributes(deterministic=True, decode=True)
    for _path, m in model.iter_modules():
        if isinstance(m, HasCache):
            input_shape = (num_envs, 1 + max_decode_len * 3, embed_dim)
            m.init_cache(input_shape, dtype=dtype)


    graphdef, _, rest = nnx.split(model, nnx.Cache, ...)
    def decode(batch, cache):
        module = nnx.merge(graphdef, cache, rest)
        module.set_attributes(deterministic=True, decode=True)
        out = module(batch)
        cache = nnx.state(module, nnx.Cache)
        return out, cache

    _, cache = decode({"sink": num_envs}, nnx.state(model, nnx.Cache))

    for step in tqdm(itertools.count(start=1)):

        # predict next_action
        # [num_envs, seq_len, num_actions] -> [num_envs, num_actions]
        logits, cache = decode({"state": timestep.observation[:, None],}, cache)
        logits = logits[:, -1]
        action = jnp.argmax(logits, axis=-1)

        # query the worlds
        timestep = jax.block_until_ready(step_fn(env_params, timestep, action))

        done = np.asarray(timestep.last())
        num_episodes += done.astype(int)
        returns += np.asarray(timestep.reward)

        # relabel for the next step
        _, cache = decode({"action": action[:, None],}, cache)
        _, cache = decode({"reward": timestep.reward[:, None],}, cache)

        # log returns if done
        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[ruleset_ids[i]].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0

        # check that all goals are done
        if jnp.all(num_episodes > eval_episodes):
            break

    return eval_info


def main(
    max_decode_len: int,
    learner_path: str,
    eval_seed: int,
    num_eval_rulesets: int,
    eval_episodes: int,
):
    config_dict = json.load(open(os.path.join(learner_path, "config.json"), "r"))

    last_step = sorted(os.listdir(os.path.join(learner_path, "models")))[-1]
    train_state = dill.load(open(os.path.join(learner_path, "models", last_step), "rb"))
    model = nnx.merge(
        train_state.graphdef,
        train_state.params,
        train_state.rest,
    )

    print("Loaded trained model")

    dataset = XMiniGridADDataset(
        config_dict["dataset_kwargs"]["data_path"],
        config_dict["dataset_kwargs"]["seq_len"],
        config_dict["seeds"]["data_seed"],
    )

    (benchmark_id, env_id, train_rulesets) = dataset.trajectories_metadata
    key = jax.random.PRNGKey(eval_seed)

    benchmark = xminigrid.load_benchmark(benchmark_id)
    all_rulesets = np.array(range(benchmark.num_rulesets()))
    eval_rulesets = np.setdiff1d(all_rulesets, train_rulesets)
    eval_indexes = np.random.randint(
        low=0, high=len(eval_rulesets), size=num_eval_rulesets
    )
    eval_rulesets = eval_rulesets[eval_indexes]

    all_rulesets = eval_rulesets
    num_envs = len(all_rulesets)

    env, env_params = xminigrid.make(env_id)
    env = GymAutoResetWrapper(env)
    rng, reset_rng = jax.random.split(key)

    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, 0)))
    step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    print("Initialized environment")

    per_gpu = num_eval_rulesets // len(jax.devices())
    idx_rank = np.arange(per_gpu)
    rulesets_per_gpu = eval_rulesets[idx_rank]
    env_params_per_gpu = env_params.replace(
        ruleset=jax.vmap(benchmark.get_ruleset)(rulesets_per_gpu)
    )
    reset_rng_per_gpu = jax.random.split(reset_rng, num_envs)[idx_rank]

    eval_info = evaluate(
        step_fn=step_fn,
        reset_fn=reset_fn,
        env_params=env_params_per_gpu,
        rng_key=rng,
        reset_rng=reset_rng_per_gpu,
        model=model,
        max_decode_len=max_decode_len,
        embed_dim=config_dict["model_config"]["model_kwargs"]["embed_dim"],
        ruleset_ids=rulesets_per_gpu,
        eval_episodes=eval_episodes,
    )
    dill.dump(
        eval_info,
        open(os.path.join(learner_path, "eval_info.dill"), "wb"),
    )


if __name__ == "__main__":
    base_path = "/home/bryanpu1/projects/aaai_2026/scaling_jax/results"
    # algo_name = "xland_ad"
    # run_name = "debug-06-02-25_14_46_07-3c1bc77f-9d04-4c25-8356-67dd1919b82a"

    # algo_name = "xland_dpt"
    # run_name = "debug-06-02-25_14_47_19-f8577fb0-3fb8-4664-8982-64d5985a9961"

    algo_name = "xland_expi"
    run_name = "debug-06-02-25_14_47_55-e68410db-d131-44d5-8720-a660d26eb3f8"

    eval_seed = 42

    num_eval_rulesets = 32
    eval_episodes = 500

    max_decode_len = 512

    learner_path = os.path.join(base_path, algo_name, run_name)
    main(max_decode_len, learner_path, eval_seed, num_eval_rulesets, eval_episodes)
