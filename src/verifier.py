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
            returns = scan_monte_carlo_returns(
                reward[..., None],
                rollout_res.solution_found[..., None],
                config.gamma,
            )
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
            returns = scan_monte_carlo_meta_returns(
                reward[..., None],
                rollout_res.solution_found[..., None],
                jnp.concatenate((
                    (rollout_res.observations == reset_token_id)[:, 1:],
                    jnp.full((len(reward), 1), fill_value=-1, dtype=int),
                ), axis=-1)[..., None],
                config.in_ep_gamma,
                config.cross_ep_gamma,
            )
            return returns
    elif config.train_loss_config.mdp_type.startswith("progress_rl"):
        def _scan_discounting(
            rews: jax.Array,
            resets: jax.Array,
        ):
            def _apply_regret(
                prev_trial, transition
            ):
                rew, reset = transition
                trial = prev_trial + reset
                val = (config.gamma ** (trial - 1)) * rew
                return trial, val

            return jax.lax.scan(
                _apply_regret,
                0,
                jnp.concatenate((rews, resets), axis=-1),
                len(rews),
                reverse=False,
            )[1]

        scan_discounting = jax.vmap(
            jax.jit(_scan_discounting),
            in_axes=[0, 0],
        )

        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape
            reset = rollout_res.observations.at[
                rollout_res.observations == eos_token_id
            ].set(reset_token_id) == reset_token_id
            padded_reset = jnp.concatenate((
                jnp.ones((B, 1), dtype=bool),
                reset,
                jnp.ones((B, 1), dtype=bool),
            ), axis=-1)
            padded_reset = jax.vmap(
                lambda x: jnp.where(x, size=T + 2, fill_value=-1)[0]
            )(padded_reset)

            diffs = padded_reset[:, 1:] - padded_reset[:, :-1]
            diffs = diffs.at[diffs < 0].set(0)
            per_trial_progress = jnp.concatenate((
                jnp.zeros((B, 1), dtype=int),
                diffs[:, 1:],
            ), axis=-1)
            returns = jnp.repeat(
                per_trial_progress,
                diffs
            ).reshape((B, T + 1))
            returns = returns[:, 1:] * rollout_res.pred_mask / batch["solution_len"][:, None]

            returns = scan_discounting(
                returns[..., None],
                jnp.concatenate((
                    (rollout_res.observations == reset_token_id)[:, 1:],
                    jnp.full((len(reward), 1), fill_value=-1, dtype=int),
                ), axis=-1)[..., None],
            )
            return returns
    elif config.train_loss_config.mdp_type.startswith("traj_improvement"):
        def _scan_discounting(
            rews: jax.Array,
            resets: jax.Array,
        ):
            def _apply_regret(
                prev_trial, transition
            ):
                rew, reset = transition
                trial = prev_trial + reset
                val = (config.gamma ** (trial - 1)) * rew
                return trial, val

            return jax.lax.scan(
                _apply_regret,
                0,
                jnp.concatenate((rews, resets), axis=-1),
                len(rews),
                reverse=False,
            )[1]

        scan_discounting = jax.vmap(
            jax.jit(_scan_discounting),
            in_axes=[0, 0],
        )

        discourage_first_reset = getattr(config.train_loss_config, "discourage_first_reset", 0)

        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape
            reset = rollout_res.observations.at[
                rollout_res.observations == eos_token_id
            ].set(reset_token_id) == reset_token_id
            padded_reset = jnp.concatenate((
                jnp.ones((B, 1), dtype=bool),
                reset,
                jnp.ones((B, 1), dtype=bool),
            ), axis=-1)
            padded_reset = jax.vmap(
                lambda x: jnp.where(x, size=T + 2, fill_value=-1)[0]
            )(padded_reset)

            diffs = padded_reset[:, 1:] - padded_reset[:, :-1]
            diffs = diffs.at[diffs < 0].set(0)
            improvement = jnp.concatenate((
                jnp.zeros((B, 1), dtype=int),
                diffs[:, 1:] - discourage_first_reset - jnp.maximum.accumulate((diffs - discourage_first_reset).at[:, 0].set(0)[:, :-1], axis=-1),
            ), axis=-1)
            returns = jnp.repeat(
                improvement,
                diffs
            ).reshape((B, T + 1))
            returns = returns[:, 1:] * rollout_res.pred_mask / (batch["solution_len"][:, None] - discourage_first_reset)

            returns = scan_discounting(
                returns[..., None],
                jnp.concatenate((
                    (rollout_res.observations == reset_token_id)[:, 1:],
                    jnp.full((len(reward), 1), fill_value=-1, dtype=int),
                ), axis=-1)[..., None],
            )

            return returns
    elif config.train_loss_config.mdp_type.startswith("traj_improvement-zero_on_reset"):
        def _scan_discounting(
            rews: jax.Array,
            resets: jax.Array,
        ):
            def _apply_regret(
                prev_trial, transition
            ):
                rew, reset, next_reset = transition
                trial = prev_trial + reset
                val = (config.gamma ** (trial - 1)) * rew * (1 - next_reset)
                return trial, val

            return jax.lax.scan(
                _apply_regret,
                0,
                jnp.concatenate((rews, resets[:-1], resets[1:]), axis=-1),
                len(rews),
                reverse=False,
            )[1]

        scan_discounting = jax.vmap(
            jax.jit(_scan_discounting),
            in_axes=[0, 0],
        )

        def process_reward(batch, rollout_res, reward, has_eos):
            B, T = rollout_res.observations.shape
            reset = rollout_res.observations.at[
                rollout_res.observations == eos_token_id
            ].set(reset_token_id) == reset_token_id
            padded_reset = jnp.concatenate((
                jnp.ones((B, 1), dtype=bool),
                reset,
                jnp.ones((B, 1), dtype=bool),
            ), axis=-1)
            padded_reset = jax.vmap(
                lambda x: jnp.where(x, size=T + 2, fill_value=-1)[0]
            )(padded_reset)

            diffs = padded_reset[:, 1:] - padded_reset[:, :-1]
            diffs = diffs.at[diffs < 0].set(0)
            improvement = jnp.concatenate((
                jnp.zeros((B, 1), dtype=int),
                diffs[:, 1:] - 1 - jnp.maximum.accumulate((diffs - 1).at[:, 0].set(0)[:, :-1], axis=-1),
            ), axis=-1)
            returns = jnp.repeat(
                improvement,
                diffs
            ).reshape((B, T + 1))
            returns = returns[:, 1:] * rollout_res.pred_mask / (batch["solution_len"][:, None] - 1)

            returns = scan_discounting(
                returns[..., None],
                jnp.concatenate((
                    jnp.concatenate((
                        (rollout_res.observations == reset_token_id)[:, 1:],
                        jnp.zeros((len(reward), 1), dtype=bool)
                    ), axis=1),
                    jnp.full((len(reward), 1), fill_value=-1, dtype=int),
                ), axis=-1)[..., None],
            )

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
