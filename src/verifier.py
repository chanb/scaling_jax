import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flax import nnx
from types import SimpleNamespace
from typing import Any, Dict

import jax
import jax.random as jrandom
import numpy as np
import timeit

from src.constants import *
from src.dataset import get_data_loader
from src.decoding import make_autoregressive
from src.learners.learner import (
    Learner,
    l2_norm,
    gather_learning_rate,
)
from src.rollout import rollout
from src.utils import parse_dict


def make_compute_returns(config, eos_token=4):
    if getattr(config.dataset_kwargs, "predict_eos", False):
        def process_target(target):
            target = "".join(np.array(target[target != eos_token]).astype(str))
            return target + "4"
        
        def get_success(response, target, mask):
            # XXX: Currently look at the first <EOS>
            if eos_token in response:
                response = "".join(np.array(
                    response[:np.where(response == eos_token)[0][0] + 1]
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
            target = "".join(np.array(target[target != eos_token]).astype(str))
            return target
        
        def get_success(response, target, mask):
            # XXX: Stop at first matching string
            success = float(target in response)

            assert "4" not in response

            if success:
                end_idx = response.find(target) + len(target)
                mask[end_idx:] = 0
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
            group_changes = np.arange(0, len(batch["sequence"]), num_rollouts_per_sample)
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
            returns = np.zeros(batch["sequence"].shape)
            for sample_i, (reward, mask, response_length) in enumerate(zip(
                rewards, batch["pred_mask"], response_lengths
            )):
                returns[sample_i][np.where(mask)[0]] = (
                    (config.gamma ** np.arange(response_length)[::-1]) * reward
                ) - (1 - has_eos[sample_i])
            return returns
    elif config.train_loss_config.mdp_type == "bandit":
        def process_reward(batch, rewards, response_lengths, has_eos):
            returns = config.gamma ** (response_lengths - 1) * (rewards - (1 - has_eos))
            return returns
    else:
        raise NotImplementedError
    

    def compute_returns(batch, eos_mask, is_prompt_mask, is_eval):
        """
        Compute verifiable rewards
        Assume each token is an action, the state is the sequence up to this point
        The reward is based on whether there is a regex match with the target

        TODO: Entropy regularization objective
        """

        response_lengths = np.zeros(batch["sequence"].shape[0])
        successes = np.zeros(batch["sequence"].shape[0])
        has_eos = np.zeros(batch["sequence"].shape[0])

        batch["pred_mask"] = 1 - np.logical_or(eos_mask, is_prompt_mask)
        eos_mask = 1 - eos_mask
        batch["first_eos_mask"] = eos_mask - np.roll(eos_mask, -1, axis=1) * eos_mask

        # Get whether or not target is in the response---neglects everything after first <EOS>
        for sample_i, (response, target, mask) in enumerate(
            zip(batch["sequence"], batch["target"], batch["pred_mask"])
        ):
            target = process_target(target)

            success, response_length, curr_has_eos, mask = get_success(
                response,
                target,
                mask,
            )
            batch["pred_mask"][sample_i] = mask
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
