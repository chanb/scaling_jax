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
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import timeit

from src.constants import *
from src.decoding import make_autoregressive
from src.learners.learner import (
    l2_norm,
    gather_learning_rate,
)
from src.learners.ppo import PPO
from src.rollout import rollout


@nnx.jit
def compute_log_probs(graphdef, params, rest, batch):
    observations = batch["observations"][:, :-1]
    actions = batch["actions"][:, 1:]

    model = nnx.merge(graphdef, params, rest)
    model.set_attributes(deterministic=False, decode=False)
    logits = model({"sequence": observations})

    lprobs = jnp.sum(
        nn.one_hot(actions, num_classes=logits.shape[-1]) * logits, axis=-1
    ) - nn.logsumexp(logits, axis=-1)

    return lprobs

class MetastablePPO(PPO):
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)

        def _sample_reset_idxes(obss, rlengths, rngs):
            reset_idxes = jnp.where(
                obss == self._dataset.reset_token_id,
                size=len(obss),
                fill_value=-1,
            )[0]
            first_reset_idx = reset_idxes[0]

            reset_idxes = jax.lax.select(
                reset_idxes >= first_reset_idx + rlengths,
                jnp.full_like(reset_idxes, fill_value=-1),
                reset_idxes,
            )

            logits = jax.lax.select(
                reset_idxes == -1,
                -jnp.inf,
                0.0,
            )
            reset_idx_i = jrandom.categorical(rngs, logits[1:])
            reset_idx = jax.lax.select(
                jnp.logical_or(
                    reset_idxes[reset_idx_i + 1] == -1,
                    jrandom.uniform(rngs) < self._config.metastable_aug_p,
                ),
                reset_idxes[0],
                reset_idxes[reset_idx_i + 1],
            )
            return reset_idx, first_reset_idx

        def _augment_sample(
            out_obss,
            out_acts,
            out_masks,
            out_rets,
            first_reset_idx,
            reset_idx,
        ):
            delta = reset_idx - first_reset_idx
            delta = jax.lax.select(
                delta <= 0,
                len(out_obss),
                -delta,
            )

            # Given [X_1, ..., X_T, <EQUAL>, ...]
            # Want a mask where <EQUAL> and onward are set to 0.
            first_reset_mask = jnp.zeros_like(out_obss)
            first_reset_mask = first_reset_mask.at[first_reset_idx].set(1)
            question_mask = 1 - jnp.cumsum(first_reset_mask)

            # Given [X_1, ..., X_T, <EQUAL>, ..., <EQUAL_N>, ...]
            # Want a mask where <EQUAL_N> and onward are set to 1.
            reset_mask = jnp.zeros_like(out_obss)
            reset_mask = reset_mask.at[reset_idx].set(1)
            reset_mask = jnp.cumsum(reset_mask)

            # Let delta = position of <EQUAL_N> - position of <EQUAL>.
            # Want a mask where the last delta entries are set to 1
            # If delta = 0, then all entries are set to 0.
            eos_mask = jnp.zeros(out_obss.shape[0])
            eos_mask = jax.lax.select(
                delta == len(out_obss),
                eos_mask,
                eos_mask.at[delta].set(1),
            )
            eos_mask = jnp.cumsum(eos_mask)

            out_obss = question_mask * out_obss + jnp.roll(reset_mask * out_obss, delta)
            out_obss = (1 - eos_mask) * out_obss + eos_mask * self._dataset.eos_token_id

            out_acts = question_mask * out_acts + jnp.roll(reset_mask * out_acts, delta)

            out_masks = question_mask * out_masks + jnp.roll(reset_mask * out_masks, delta)
            out_masks = (1 - eos_mask) * out_masks

            out_rets = question_mask * out_rets + jnp.roll(reset_mask * out_rets, delta)
            out_rets = (1 - eos_mask) * out_rets

            return (
                out_obss.astype(int),
                out_acts.astype(int),
                out_masks.astype(int),
                out_rets.astype(int),
            )
        
        def _identity(
            out_obss,
            out_acts,
            out_masks,
            out_rets,
            first_reset_idx,
            reset_idx,
        ):
            return (
                out_obss.astype(int),
                out_acts.astype(int),
                out_masks.astype(int),
                out_rets.astype(int),
            )

        def _augment(iter_i, state):
            obss = state["observations"][iter_i]
            acts = state["actions"][iter_i]
            masks = state["pred_mask"][iter_i]
            rets = state["returns"][iter_i]
            first_reset_idx = state["first_reset_idx"][iter_i]
            reset_idx = state["reset_idx"][iter_i]
            success = state["successes"][iter_i]
            out_obss = jnp.copy(obss)
            out_acts = jnp.copy(acts)
            out_masks = jnp.copy(masks)
            out_rets = jnp.copy(rets)
            (
                out_obss,
                out_acts,
                out_masks,
                out_rets,
            ) = jax.lax.cond(
                success,
                _augment_sample,
                _identity,
                out_obss,
                out_acts,
                out_masks,
                out_rets,
                first_reset_idx,
                reset_idx,
            )

            return {
                "observations": state["observations"].at[iter_i].set(out_obss),
                "actions": state["actions"].at[iter_i].set(out_acts),
                "pred_mask": state["pred_mask"].at[iter_i].set(out_masks),
                "returns": state["returns"].at[iter_i].set(out_rets),
                "first_reset_idx": state["first_reset_idx"],
                "reset_idx": state["reset_idx"],
                "successes": state["successes"],
            }
        self._sample_reset_idxes = jax.vmap(_sample_reset_idxes)
        self._augment = jax.jit(_augment)

    def update(self, epoch: int, *args, **kwargs) -> Dict[str, Any]:
        curr_rng = jrandom.fold_in(self._rng, epoch)

        auxes = []
        total_sample_time = 0
        total_update_time = 0
        total_rollout_time = 0

        for update_i in range(self._num_updates_per_epoch):
            curr_rng = jrandom.fold_in(curr_rng, update_i)

            # Sample batch of questions/prompts
            tic = timeit.default_timer()
            batch = self.get_batch()

            batch["sequence"] = np.repeat(
                batch["sequence"],
                self.num_rollouts_per_sample,
                axis=0,
            )
            batch["mask"] = np.repeat(
                batch["mask"],
                self.num_rollouts_per_sample,
                axis=0,
            )
            batch["target"] = np.repeat(
                batch["target"],
                self.num_rollouts_per_sample,
                axis=0,
            )
            batch["pointer_correct"] = np.repeat(
                batch["pointer_correct"],
                self.num_rollouts_per_sample,
                axis=0,
            )

            total_sample_time += timeit.default_timer() - tic

            # Sample rollouts
            tic = timeit.default_timer()
            module = nnx.merge(self.state.graphdef, self.state.params, self.state.rest)
            _, init_cache = make_autoregressive(
                module,
                max_decode_len=batch["sequence"].shape[1],
                batch_size=batch["sequence"].shape[0],
                embed_dim=self._config.model_config.model_kwargs.embed_dim,
                dtype=self.dtype,
                eval_mode=True,
            )
            cache = init_cache()
            graphdef, _, rest = nnx.split(module, nnx.Cache, ...)
            (observations, actions, _, _, last_prompt_idxes) = rollout(
                graphdef,
                cache,
                rest,
                curr_rng,
                batch,
                eos_token=self._dataset.eos_token_id,
                correct_aware_shift=getattr(self._dataset, "correctness_aware_tokens_offset", 0),
                max_token_id_to_shift=getattr(self._dataset, "max_token_id_to_shift", 0),
            )

            # Compute return
            batch["observations"] = observations
            batch["actions"] = actions
            returns, successes, response_lengths = self._compute_returns(
                batch,
                last_prompt_idxes,
                is_eval=False,
            )
            batch["returns"] = returns

            total_rollout_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            # Split intermediate rollouts
            if (
                self._config.metastable_aug_p > 0.0
                and hasattr(self._dataset, "reset_token_id")
                and np.any(successes)
            ):
                aug_rngs = jrandom.split(curr_rng, len(batch["observations"]))
                
                reset_idx, first_reset_idx = self._sample_reset_idxes(
                    batch["observations"],
                    response_lengths,
                    aug_rngs,
                )

                aug_data = jax.lax.fori_loop(
                    0,
                    len(batch["observations"]),
                    self._augment,
                    {
                        "observations": batch["observations"],
                        "actions": batch["actions"],
                        "pred_mask": batch["pred_mask"],
                        "returns": batch["returns"],
                        "first_reset_idx": first_reset_idx,
                        "reset_idx": reset_idx,
                        "successes": successes,
                    }
                )

                (
                    batch["observations"],
                    batch["actions"],
                    batch["pred_mask"],
                    batch["returns"],
                ) = (
                    aug_data["observations"],
                    aug_data["actions"],
                    aug_data["pred_mask"],
                    aug_data["returns"],
                )

            # Compute log probs
            batch["old_lprobs"] = compute_log_probs(
                self.state.graphdef,
                self.state.params,
                self.state.rest,
                batch,
            )

            for update_i in range(self._config.num_ppo_steps):
                self._state, aux = self.train_step(
                    self._state,
                    batch,
                )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS].item()), f"Loss became NaN\naux: {aux}"

            aux[CONST_TRAIN][CONST_SUCCESS_RATE] = np.mean(successes)
            aux[CONST_TRAIN][CONST_RESPONSE_LENGTH] = np.mean(response_lengths)

            auxes.append(aux)

        auxes = jax.tree_util.tree_map(
            lambda *args: np.mean([np.asarray(el) for el in args]),
            *auxes,
        )

        log = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AGG_LOSS].item(),
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"time/{CONST_ROLLOUT_TIME}": total_rollout_time,
            f"{CONST_GRAD_NORM}/model": auxes[CONST_GRAD_NORM][CONST_MODEL].item(),
            f"{CONST_PARAM_NORM}/model": l2_norm(self._state.params).item(),
            **{
                f"train/{k}": v for k, v in auxes[CONST_TRAIN].items()
            },
            **{
                f"hist/{k}": v for k, v in aux[CONST_HIST].items()
            },
        }

        if isinstance(self._state.opt_state, dict):
            for model_name, optimizer in self._state.opt_state:
                gather_learning_rate(aux, model_name, optimizer)
        else:
            gather_learning_rate(aux, CONST_MODEL, self._state.opt_state)
        return log
