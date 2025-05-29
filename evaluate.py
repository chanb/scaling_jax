import dill
import itertools
import jax
import jax.numpy as jnp
import jax.random as jrandom
import json
import numpy as np
import os
import xminigrid

from flax import nnx

from xminigrid.core.constants import NUM_ACTIONS
from xminigrid.environment import EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from src.datasets.ad_dataset import XMiniGridADDataset


def evaluate(
    step_fn,
    reset_fn,
    env_params: EnvParams,
    rng_key: jax.Array,
    reset_rng: jax.Array,
    model,
    ruleset_ids: np.ndarray,
    eval_episodes: int,
):
    num_envs = len(ruleset_ids)
    kv_cache = model.init_cache(
        batch_size=num_envs, dtype=torch.float16, device=model.device
    )

    num_episodes = np.zeros(num_envs)
    returns = np.zeros(num_envs)

    eval_info = defaultdict(list)

    with jax.default_device(jax.devices("cuda")[rank]):
        timestep = jax.block_until_ready(reset_fn(env_params, reset_rng))
        prev_action, prev_reward = jnp.zeros(num_envs), jnp.zeros(num_envs)

    for step in itertools.count(start=1):

        # predict next_action
        # [num_envs, seq_len, num_actions] -> [num_envs, num_actions]
        logits = model(
            {
                "state": timestep.observation[:, None],
                "action": prev_action[:, None],
                "reward": prev_reward[:, None],
            }
        )
        logits = logits[:, -1]
        dist = jrandom.categorical(
            jrandom.foldin(rng_key, step), logits, axis=-1,
        )
        action = jnp.argmax(logits, axis=-1)

        # query the worlds
        timestep = jax.block_until_ready(step_fn(env_params, timestep, action_jnp))

        done = np.asarray(timestep.last())
        num_episodes += done.astype(int)
        returns += np.asarray(timestep.reward)

        # relabel for the next step
        prev_action = action_jnp
        prev_reward = timestep.reward

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


def main(learner_path: str, eval_seed: int, num_eval_rulesets: int, eval_episodes: int):
    config_dict = json.load(open(os.path.join(learner_path, "config.json"), "r"))
    train_state = dill.load(open(os.path.join(learner_path, "models", "00000.dill"), "rb"))
    model = nnx.merge(
        train_state.graphdef,
        train_state.params,
        train_state.rest,
    )

    model.eval()
    model.set_attributes(deterministic=True, decode=True)
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
    idx_rank = np.arange(per_gpu * config.local_rank, per_gpu)
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
        model=model_engine,
        ruleset_ids=rulesets_per_gpu,
        eval_episodes=eval_episodes,
    )
    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    base_path = "/home/chanb/scratch/results"
    algo_name = "xland_ad"
    run_name = "debug-05-29-25_11_08_17-f08106e8-f9d2-4503-808e-04a3afa8325d"
    eval_seed = 42

    num_eval_rulesets = 128
    eval_episodes = 100

    learner_path = os.path.join(base_path, algo_name, run_name)
    main(learner_path, eval_seed, num_eval_rulesets, eval_episodes)
