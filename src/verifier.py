import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import jax
import jax.numpy as jnp
import numpy as np

from src.constants import *


def make_compute_returns(config, eos_token_id, reset_token_id, token_map):
    # Reward shaping
    reward_type = getattr(config, "reward_type", "default")
    if reward_type == "negative_on_failure":
        def shape_reward(batch, rollout_res):
            reward = jnp.zeros_like(rollout_res.actions)
            reward = reward.at[
                jnp.arange(len(rollout_res.response_length)),
                batch["question_len"] + rollout_res.response_length - 1
            ].set((-1) ** (1 - rollout_res.success))
            return reward

        def bandit_reward(rollout_res):
            return (-1) ** (1 - rollout_res.success)
    elif reward_type == "negative_dense":
        def shape_reward(batch, rollout_res):
            reward = jnp.full_like(rollout_res.actions, fill_value=-1)
            reward = reward.at[
                jnp.arange(len(rollout_res.response_length)),
                batch["question_len"] + rollout_res.response_length - 1
            ].set(rollout_res.success - 1)
            return reward

        def bandit_reward(rollout_res):
            return rollout_res.success - 1
    else:
        def shape_reward(batch, rollout_res):
            reward = jnp.zeros_like(rollout_res.actions)
            reward = reward.at[
                jnp.arange(len(rollout_res.response_length)),
                batch["question_len"] + rollout_res.response_length - 1
            ].set(rollout_res.success)
            return reward

        def bandit_reward(rollout_res):
            return rollout_res.success

    # Dr. GRPO
    dr_grpo = getattr(config, "dr_grpo", False)
    batch_size = getattr(config, "batch_size", 1)
    num_rollouts_per_sample = getattr(config, "num_rollouts_per_sample", 1)
    if dr_grpo and num_rollouts_per_sample > 1:
        def normalize_reward(rewards):
            group_changes = np.arange(0, batch_size, num_rollouts_per_sample)
            group_means = np.add.reduceat(rewards, group_changes) / num_rollouts_per_sample
            group_means = np.repeat(group_means, num_rollouts_per_sample, axis=0)
            rewards = rewards - group_means
            return rewards
    else:
        def normalize_reward(rewards):
            return rewards

    # MDP vs Bandit formulation
    if config.train_loss_config.mdp_type.startswith("episodic"):
        def _scan_monte_carlo_returns(
            rews: jax.Array,
            dones: jax.Array,
            gamma: float,
        ):
            def _returns(
                next_val, transition
            ):
                rew, done = transition
                val = (next_val * gamma) * (1 - done) + rew
                return val, val

            return jax.lax.scan(
                _returns,
                0,
                jnp.concatenate((rews, dones), axis=-1),
                len(rews),
                reverse=True,
            )[1]

        scan_monte_carlo_returns = jax.vmap(
            jax.jit(_scan_monte_carlo_returns),
            in_axes=[0, 0, None],
        )

        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape

            remainder = (T - 1) % config.attempt_length
            explore_mask = jnp.roll(rollout_res.solution_found, 1, axis=-1)

            returns = scan_monte_carlo_returns(
                reward[..., None],
                rollout_res.solution_found[..., None],
                config.gamma,
            ) * jnp.logical_and(
                rollout_res.pred_mask,
                jnp.logical_not(explore_mask),
            )
            end_idx = T - remainder
            obss = rollout_res.observations[:, 1:end_idx].reshape((B, -1, config.attempt_length))
            num_trials = obss.shape[1]
            if config.train_loss_config.in_rollout_traj:
                # # Unique trajs in single rollout
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, None]) ** 2, axis=-1)
                pairwise_traj_diff = jnp.tril(pairwise_traj_diff) - jnp.triu(jnp.ones_like(pairwise_traj_diff))
                all_diff = jnp.repeat(
                    (
                        jnp.logical_not(jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True))
                        # / (jnp.arange(num_trials) + 1)[:, None]
                    ),
                    config.attempt_length,
                    axis=-1,
                ).reshape((B, -1))
                all_diff = jnp.concatenate((jnp.zeros((B, 1)), all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + all_diff * explore_mask

            if config.train_loss_config.cross_rollout_traj:
                # Unique trajs across parallel rollouts
                obss = obss.reshape(-1, config.num_rollouts_per_sample, num_trials, config.attempt_length)
                obss = jnp.transpose(obss, (0, 2, 1, 3))
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, :, :, None]) ** 2, axis=-1)
                pairwise_traj_diff = pairwise_traj_diff - np.eye(config.num_rollouts_per_sample)[None, None]
                parallel_all_diff = jnp.logical_not(jnp.repeat(
                    jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True),
                    config.attempt_length,
                    axis=-1,
                ))
                parallel_all_diff = jnp.transpose(parallel_all_diff, (0, 2, 1, 3)).reshape((B, -1))
                parallel_all_diff = jnp.concatenate((jnp.zeros((B, 1)), parallel_all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + parallel_all_diff * explore_mask
            return returns
    elif config.train_loss_config.mdp_type.startswith("meta_rl"):
        # TODO: Use in-hindsight reward fraction
        def _scan_monte_carlo_meta_returns(
            rews: jax.Array,
            dones: jax.Array,
            resets: jax.Array,
            in_ep_gamma: float,
            cross_ep_gamma: float,
        ):
            def _returns(
                next_val, transition
            ):
                rew, done, reset = transition
                val = (
                    (1 - reset) * next_val * in_ep_gamma
                    + reset * next_val * cross_ep_gamma
                ) * (1 - done) + rew
                return val, val

            return jax.lax.scan(
                _returns,
                0,
                jnp.concatenate((rews, dones, resets), axis=-1),
                len(rews),
                reverse=True,
            )[1]

        scan_monte_carlo_meta_returns = jax.vmap(
            jax.jit(_scan_monte_carlo_meta_returns),
            in_axes=[0, 0, 0, None, None],
        )

        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape

            remainder = (T - 1) % config.attempt_length
            explore_mask = jnp.roll(rollout_res.solution_found, 1, axis=-1)

            returns = scan_monte_carlo_meta_returns(
                reward[..., None],
                rollout_res.solution_found[..., None],
                jnp.concatenate((
                    (rollout_res.observations == reset_token_id)[:, 1:],
                    jnp.full((len(reward), 1), fill_value=-1, dtype=int),
                ), axis=-1)[..., None],
                config.in_ep_gamma,
                config.cross_ep_gamma,
            ) * jnp.logical_and(
                rollout_res.pred_mask,
                jnp.logical_not(explore_mask),
            )

            end_idx = T - remainder
            obss = rollout_res.observations[:, 1:end_idx].reshape((B, -1, config.attempt_length))
            num_trials = obss.shape[1]
            if config.train_loss_config.in_rollout_traj:
                # # Unique trajs in single rollout
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, None]) ** 2, axis=-1)
                pairwise_traj_diff = jnp.tril(pairwise_traj_diff) - jnp.triu(jnp.ones_like(pairwise_traj_diff))
                all_diff = jnp.repeat(
                    (
                        jnp.logical_not(jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True))
                        # / (jnp.arange(num_trials) + 1)[:, None]
                    ),
                    config.attempt_length,
                    axis=-1,
                ).reshape((B, -1))
                all_diff = jnp.concatenate((jnp.zeros((B, 1)), all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + all_diff * explore_mask

            if config.train_loss_config.ordered_traj and config.attempt_length == 2:
                expected_order = np.vstack((
                    np.full(config.dataset_kwargs.vocab_size, fill_value=config.dataset_kwargs.vocab_size),
                    np.arange(config.dataset_kwargs.vocab_size)
                )).T
                match_order = ((obss == expected_order[None])[..., ::-1]).reshape((B, -1))
                match_order = jnp.concatenate((jnp.zeros((B, 1)), match_order, jnp.zeros((B, remainder))), axis=1)
                returns = returns + match_order * explore_mask


            if config.train_loss_config.cross_rollout_traj:
                # Unique trajs across parallel rollouts
                obss = obss.reshape(-1, config.num_rollouts_per_sample, num_trials, config.attempt_length)
                obss = jnp.transpose(obss, (0, 2, 1, 3))
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, :, :, None]) ** 2, axis=-1)
                pairwise_traj_diff = pairwise_traj_diff - np.eye(config.num_rollouts_per_sample)[None, None]
                parallel_all_diff = jnp.logical_not(jnp.repeat(
                    jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True),
                    config.attempt_length,
                    axis=-1,
                ))
                parallel_all_diff = jnp.transpose(parallel_all_diff, (0, 2, 1, 3)).reshape((B, -1))
                parallel_all_diff = jnp.concatenate((jnp.zeros((B, 1)), parallel_all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + parallel_all_diff * explore_mask
            return returns
    elif config.train_loss_config.mdp_type.startswith("traj_improvement"):
        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape

            remainder = (T - 1) % config.attempt_length
            end_idx = T - remainder

            explore_mask = jnp.roll(rollout_res.solution_found, 1, axis=-1)

            obss = rollout_res.observations[:, 1:end_idx].reshape((B, -1, config.attempt_length))
            match_soln = (obss == batch["target"][:, :config.attempt_length][:, None])
            match_len = jnp.sum(jnp.minimum.accumulate(match_soln, axis=-1), axis=-1, keepdims=True) - 1
            match_len = jnp.concatenate((jnp.zeros((B, 1, 1)), match_len), axis=1)
            diff = (match_len[:, 1:] - match_len[:, :-1]) / (config.attempt_length - 1)
            diff = jnp.repeat(diff, config.attempt_length, axis=-1).reshape((B, -1))
            diff = jnp.concatenate((jnp.zeros((B, 1)), diff, jnp.zeros((B, remainder))), axis=1)
            returns = diff * jnp.logical_and(
                rollout_res.pred_mask,
                jnp.logical_not(explore_mask),
            )

            if config.train_loss_config.in_rollout_traj:
                # # Unique trajs in single rollout
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, None]) ** 2, axis=-1)
                pairwise_traj_diff = jnp.tril(pairwise_traj_diff) - jnp.triu(jnp.ones_like(pairwise_traj_diff))
                all_diff = jnp.logical_not(jnp.repeat(
                    jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True),
                    config.attempt_length,
                    axis=-1,
                ).reshape((B, -1)))
                all_diff = jnp.concatenate((jnp.zeros((B, 1)), all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + all_diff * explore_mask

            if config.train_loss_config.cross_rollout_traj:
                # Unique trajs across parallel rollouts
                num_trials = obss.shape[1]
                obss = obss.reshape(-1, config.num_rollouts_per_sample, num_trials, config.attempt_length)
                obss = jnp.transpose(obss, (0, 2, 1, 3))
                pairwise_traj_diff = jnp.sum((obss[:, :, None] - obss[:, :, :, None]) ** 2, axis=-1)
                pairwise_traj_diff = pairwise_traj_diff - np.eye(config.num_rollouts_per_sample)[None, None]
                parallel_all_diff = jnp.logical_not(jnp.repeat(
                    jnp.max(pairwise_traj_diff == 0, axis=-1, keepdims=True),
                    config.attempt_length,
                    axis=-1,
                ))
                parallel_all_diff = jnp.transpose(parallel_all_diff, (0, 2, 1, 3)).reshape((B, -1))
                parallel_all_diff = jnp.concatenate((jnp.zeros((B, 1)), parallel_all_diff, jnp.zeros((B, remainder))), axis=1)
                returns = returns + parallel_all_diff * explore_mask

            return returns
    elif config.train_loss_config.mdp_type == "bandit":
        def process_reward(batch, rollout_res, reward, has_eos):
            returns = config.gamma ** (rollout_res.response_length - 1) * (bandit_reward(rollout_res) - (1 - has_eos))
            return returns
    else:
        raise NotImplementedError
    

    def compute_returns(batch, rollout_res):
        """
        Compute verifiable rewards
        Assume each token is an action, the state is the sequence up to this point
        The reward is based on whether there is a regex match with the target

        TODO: Entropy regularization objective
        """
        has_eos = jnp.max(jnp.logical_or(
            rollout_res.solution_found,
            rollout_res.eos,
        ), axis=-1)
        
        has_eos = jnp.ones_like(has_eos, dtype=bool)

        reward = shape_reward(batch, rollout_res)
        reward = normalize_reward(reward)
        returns = process_reward(batch, rollout_res, reward, has_eos)

        return returns
    return compute_returns
