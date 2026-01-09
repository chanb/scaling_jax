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
    if getattr(config.dataset_kwargs, "predict_eos", True):
        def process_target(target):
            target = "".join(np.array(target[target != eos_token_id]).astype(str))
            return target + str(eos_token_id)
        
        def get_success(response, target, mask):
            # XXX: Currently look at the first <EOS>
            if eos_token_id in response:
                response = "".join(np.array(
                    response[:np.where(response == eos_token_id)[0][0] + 1]
                ).astype(str))
                has_eos = 1.0
            else:
                response = "".join(np.array(response).astype(str))
                has_eos = 0.0

            response_length = np.sum(mask)
            success = float(target in response)
            return success, response_length, has_eos, mask
    else:
        def process_target(target):
            target = "".join(np.array(target[target != eos_token_id]).astype(str))
            return target
        
        def get_success(response, target, mask):
            # XXX: Stop at first matching string
            response = "".join(np.array(response).astype(str))
            success = float(target in response)

            assert str(eos_token_id) not in response

            if success:
                end_idx = response.find(target) + len(target) - 1
                mask[end_idx:] = 0

            # print(success, target, response)
            response_length = np.sum(mask)

            has_eos = 1.0
            return success, response_length, has_eos, mask

    # Reward shaping
    reward_type = getattr(config, "reward_type", "default")
    if reward_type == "negative_on_failure":
        def shape_reward(successes):
            return (-1) ** (1 - successes)
    elif reward_type == "negative_dense":
        def shape_reward(successes):
            return successes - 1
    else:
        def shape_reward(successes):
            return successes

    # Dr. GRPO
    dr_grpo = getattr(config, "dr_grpo", False)
    num_rollouts_per_sample = getattr(config, "num_rollouts_per_sample", 1)
    if dr_grpo and num_rollouts_per_sample > 1:
        def normalize_reward(batch, rewards):
            group_changes = np.arange(0, len(batch["observations"]), num_rollouts_per_sample)
            group_means = np.add.reduceat(rewards, group_changes) / num_rollouts_per_sample
            group_means = np.repeat(group_means, num_rollouts_per_sample, axis=0)
            rewards = rewards - group_means
            return rewards
    else:
        def normalize_reward(batch, rewards):
            return rewards

    # MDP vs Bandit formulation
    if config.train_loss_config.mdp_type.startswith("episodic"):
        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = np.zeros(batch["observations"].shape)
            for sample_i, (reward, mask, response_length) in enumerate(zip(
                rewards, batch["pred_mask"], response_lengths
            )):
                returns[sample_i][np.where(mask)[0]] = (
                    (config.gamma ** np.arange(response_length)[::-1]) * reward
                ) - (1 - has_eos[sample_i])
            return returns
    elif config.train_loss_config.mdp_type.startswith("multiturn"):
        # TODO: Use in-hindsight reward fraction
        @jax.jit
        def scan_fn(carry, act_idx):
            """Scan function that processes indices in reverse order."""
            returns_row = carry["returns"]
            
            # Check if this is the last action (first in reversed order)
            is_last = (act_idx == carry["last_act_idx"])
            
            # Calculate return for non-last actions
            is_reset = carry["is_reset"][act_idx]
            gamma = (1 - is_reset) * config.gamma + is_reset * config.reset_gamma
            non_last_return = gamma * returns_row[act_idx + 1]
            
            # Select based on whether this is the last action
            new_return = jnp.where(is_last, carry["terminal_reward"], non_last_return)
            
            # Update returns
            returns_row = returns_row.at[act_idx].set(new_return)
            
            return {
                "returns": returns_row,
                "is_reset": carry["is_reset"],
                "terminal_reward": carry["terminal_reward"],
                "last_act_idx": carry["last_act_idx"],
            }, None

        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = np.zeros(batch["observations"].shape)
            for sample_i, (reward, mask, actions, response_length) in enumerate(zip(
                rewards, batch["pred_mask"], batch["actions"], response_lengths
            )):
                last_return = reward - (1 - has_eos[sample_i])
                is_resets = (actions == reset_token_id).astype(jnp.float32)
                act_idxes = np.where(mask)[0][::-1]

                # Initialize with current returns for this sample
                init_returns = {
                    "returns": returns[sample_i],
                    "is_reset": is_resets,
                    "terminal_reward": last_return,
                    "last_act_idx": act_idxes[0],
                }

                # Run scan over reversed indices
                final_returns, _ = jax.lax.scan(scan_fn, init_returns, act_idxes)

                # Update the returns array
                returns[sample_i] = final_returns["returns"]
            return returns
    elif config.train_loss_config.mdp_type.startswith("traj_improvement"):
        @jax.jit
        def scan_fn(carry, idx):
            pointer_correct = carry["pointer_correct"]
            actions = carry["actions"]
            target = carry["target"]
            pred_mask = carry["pred_mask"]
            last_reset_idx = carry["last_reset_idx"]
            first_mistake_idx = carry["first_mistake_idx"]
            reset_idxes = carry["reset_idxes"]
            first_mistake_idxes = carry["first_mistake_idxes"]
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
                pred_mask[idx],
                jax.lax.select(
                    action_match,
                    pointer_correct + 1,
                    reset_pointer,
                ),
                pointer_correct,
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
            first_mistake_idxes = first_mistake_idxes.at[curr_trial].set(first_mistake_idx)

            # Identify the first mistake index within the current trial
            first_mistake_idx = jax.lax.select(
                jnp.logical_or(action_match, is_reset),
                idx + 1,
                jax.lax.select(
                    first_mistake_idx > reset_idxes[curr_trial],
                    first_mistake_idx,
                    idx,
                ),
            )

            curr_trial = jax.lax.select(
                is_reset_with_pred,
                curr_trial + 1,
                curr_trial,
            )

            return {
                "pointer_correct": pointer_correct,
                "last_reset_idx": last_reset_idx,
                "first_mistake_idx": first_mistake_idx,
                "actions": actions,
                "target": target,
                "pred_mask": pred_mask,
                "reset_idxes": reset_idxes,
                "first_mistake_idxes": first_mistake_idxes,
                "curr_trial": curr_trial,
            }, None

        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = np.zeros(batch["observations"].shape)
            for sample_i, (pred_mask, actions, target) in enumerate(zip(
                batch["pred_mask"], batch["actions"], batch["target"]
            )):
                pointer_correct = np.array(1, dtype=int)
                last_reset_idx = np.array(-1, dtype=int)
                first_mistake_idx = np.array(-1, dtype=int)
                curr_trial = np.array(0, dtype=int)
                reset_idxes = np.full_like(actions, fill_value=-1, dtype=int)
                reset_idxes[0] = np.where(pred_mask == 1)[0][0] - 1
                first_mistake_idxes = np.full_like(actions, fill_value=-1, dtype=int)
                last_idx = min(np.where(pred_mask == 1)[0][-1] + 1, actions.shape[-1])

                res, _ = jax.lax.scan(
                    scan_fn,
                    {
                        "pointer_correct": pointer_correct,
                        "last_reset_idx": last_reset_idx,
                        "first_mistake_idx": first_mistake_idx,
                        "actions": actions,
                        "target": target,
                        "pred_mask": pred_mask.astype(int),
                        "reset_idxes": reset_idxes,
                        "first_mistake_idxes": first_mistake_idxes,
                        "curr_trial": curr_trial,
                    },
                    np.arange(last_idx),
                )

                reset_idxes = res["reset_idxes"]
                first_mistake_idxes = res["first_mistake_idxes"]

                last_idx = min(np.where(pred_mask == 1)[0][-1] + 1, actions.shape[-1])
                correct_lens = np.concatenate(([0], first_mistake_idxes - reset_idxes))
                improvements = correct_lens[1:] - correct_lens[:-1]
                reset_idxes = reset_idxes.at[(np.where(reset_idxes == -1))[0][0]].set(last_idx)
                trial_lengths = np.diff(reset_idxes[reset_idxes != -1])

                # Update the returns array
                returns[sample_i, reset_idxes[0]:last_idx] = np.repeat(
                    improvements[:int(np.sum(reset_idxes != -1)) - 1], trial_lengths
                )
            return returns
    elif config.train_loss_config.mdp_type == "bandit":
        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = config.gamma ** (response_lengths - 1) * (rewards - (1 - has_eos))
            return returns
    else:
        raise NotImplementedError
    

    def compute_returns(batch, last_prompt_idxes, is_eval):
        """
        Compute verifiable rewards
        Assume each token is an action, the state is the sequence up to this point
        The reward is based on whether there is a regex match with the target

        TODO: Entropy regularization objective
        """

        response_lengths = np.zeros(batch["sequence"].shape[0])
        successes = np.zeros(batch["sequence"].shape[0])
        has_eos = np.zeros(batch["sequence"].shape[0])

        batch["pred_mask"] = np.zeros_like(batch["sequence"])

        # Get whether or not target is in the response---neglects everything after first <EOS>
        for sample_i, (obs, act, target, last_prompt_idx) in enumerate(zip(
            batch["observations"],
            batch["actions"],
            batch["target"],
            last_prompt_idxes,
        )):
            target = process_target(target)
            question_mask = np.ones(batch["sequence"].shape[-1])
            question_mask[last_prompt_idx + 1:] = 0
            answer_mask = 1 - question_mask

            pred_mask = np.zeros(batch["sequence"].shape[-1])
            pred_mask[last_prompt_idx:] = 1

            response = np.concatenate((
                obs[np.where(question_mask)],
                act[np.where(answer_mask)],
            ))

            response = np.array([token_map[int(token)] for token in response])
            # print("=" * 50)
            # print(last_prompt_idx)
            # print(question_mask)
            # print(obs)
            # print(answer_mask)
            # print(act)
            # print(response)
            success, response_length, curr_has_eos, pred_mask = get_success(
                response,
                target,
                pred_mask,
            )
            batch["pred_mask"][sample_i] = pred_mask
            successes[sample_i] = success
            response_lengths[sample_i] = response_length
            has_eos[sample_i] = curr_has_eos

        if is_eval:
            return successes, response_lengths

        rewards = shape_reward(successes)
        rewards = normalize_reward(batch, rewards)
        returns = process_reward(batch, rewards, response_lengths, has_eos)

        return returns, successes, response_lengths
    return compute_returns
