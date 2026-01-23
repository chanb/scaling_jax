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
    elif config.train_loss_config.mdp_type.startswith("traj_improvement"):
        @jax.jit
        def scan_fn(carry, idx):
            pointer_correct = carry["pointer_correct"]
            actions = carry["actions"]
            target = carry["target"]
            pred_mask = carry["pred_mask"]
            last_reset_idx = carry["last_reset_idx"]
            reset_idxes = carry["reset_idxes"]
            correct_lens = carry["correct_lens"]
            curr_trial = carry["curr_trial"]

            action_match = actions[idx] == target[pointer_correct]
            is_reset = actions[idx] == reset_token_id
            is_reset_with_pred = jnp.logical_and(is_reset, pred_mask[idx])

            # Shift the pointer if the action matches the target and we're within a prediction mask
            reset_pointer = jax.lax.select(
                is_reset,
                1,
                0,
            )
            pointer_correct = jax.lax.select(
                action_match,
                pointer_correct + 1, # Increment pointer by 1 if current token matches
                reset_pointer, # Reset to 0 if it's a mistake and not a reset token, to 1 otherwise
            )

            # Update the last reset index to current index upon new trial
            last_reset_idx = jax.lax.select(
                is_reset_with_pred,
                idx,
                last_reset_idx,
            )

            reset_idxes = reset_idxes.at[curr_trial + 1].set(
                jax.lax.select(
                    is_reset_with_pred,
                    last_reset_idx,
                    reset_idxes[curr_trial + 1],
                )
            )

            correct_lens = correct_lens.at[curr_trial].set(
                jnp.maximum(pointer_correct, correct_lens[curr_trial])
            )

            curr_trial = jax.lax.select(
                is_reset_with_pred,
                curr_trial + 1,
                curr_trial,
            )

            return {
                "pointer_correct": pointer_correct,
                "last_reset_idx": last_reset_idx,
                "actions": actions,
                "target": target,
                "pred_mask": pred_mask,
                "reset_idxes": reset_idxes,
                "correct_lens": correct_lens,
                "curr_trial": curr_trial,
            }, None

        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = np.zeros(batch["observations"].shape)
            for sample_i, (pred_mask, actions, target) in enumerate(zip(
                batch["pred_mask"], batch["actions"], batch["target"]
            )):
                pointer_correct = np.array(1, dtype=int)
                last_reset_idx = np.array(-1, dtype=int)
                curr_trial = np.array(0, dtype=int)
                reset_idxes = np.full_like(actions, fill_value=-1, dtype=int)
                reset_idxes[0] = np.where(pred_mask == 1)[0][0] - 1
                correct_lens = np.full_like(actions, fill_value=-1, dtype=int)
                last_idx = min(np.where(pred_mask == 1)[0][-1] + 2, actions.shape[-1])

                res, _ = jax.lax.scan(
                    scan_fn,
                    {
                        "pointer_correct": pointer_correct,
                        "last_reset_idx": last_reset_idx,
                        "actions": actions.at[:reset_idxes[0] + 1].set(reset_token_id),
                        "target": target,
                        "pred_mask": pred_mask.astype(int),
                        "reset_idxes": reset_idxes,
                        "correct_lens": correct_lens,
                        "curr_trial": curr_trial,
                    },
                    np.arange(last_idx),
                )

                reset_idxes = res["reset_idxes"]
                correct_lens = res["correct_lens"]
                correct_lens = np.concatenate(([0], correct_lens))

                # TODO: Negative reward
                answer_len = np.where(target == eos_token_id)[0]
                if len(answer_len) > 0:
                    answer_len = answer_len[0]
                else:
                    answer_len = len(target)

                last_idx = min(np.where(pred_mask == 1)[0][-1] + 1, actions.shape[-1])
                if getattr(config, "cumulative", True):
                    cum_correct_lens = np.maximum.accumulate(correct_lens)
                    improvements = (correct_lens[1:] - cum_correct_lens[:-1] - 1) / answer_len
                else:
                    improvements = correct_lens[1:] - correct_lens[:-1]
                reset_idxes = reset_idxes.at[(np.where(reset_idxes == -1))[0][0]].set(last_idx)
                trial_lengths = np.diff(reset_idxes[reset_idxes != -1])

                # Update the returns array
                num_trials = int(np.sum(reset_idxes != -1)) - 1
                if getattr(config, "discounting", True):
                    returns[sample_i, reset_idxes[0]:last_idx] = np.repeat(
                        config.gamma ** (
                            np.arange(num_trials)
                        ) * improvements[:num_trials],
                        trial_lengths,
                    )
                else:
                    # Regret like
                    returns[sample_i, reset_idxes[0]:last_idx] = np.repeat(
                        improvements[:num_trials] / (
                            np.arange(num_trials) + 1
                        ),
                        trial_lengths,
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
        
        if not config.dataset_kwargs.predict_eos:
            has_eos = jnp.ones_like(has_eos, dtype=bool)

        reward = shape_reward(batch, rollout_res)
        reward = normalize_reward(reward)
        returns = process_reward(batch, rollout_res, reward, has_eos)

        return returns
    return compute_returns
