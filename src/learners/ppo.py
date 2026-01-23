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
from src.learners.reinforce import REINFORCE
from src.rollout import rollout


@nnx.jit
def compute_log_probs(graphdef, params, rest, batch):
    observations = batch["observations"]
    actions = batch["actions"]

    model = nnx.merge(graphdef, params, rest)
    model.set_attributes(deterministic=False, decode=False)
    logits = model({"sequence": observations})

    lprobs = jnp.sum(
        nn.one_hot(actions, num_classes=logits.shape[-1]) * logits, axis=-1
    ) - nn.logsumexp(logits, axis=-1)

    return lprobs

class PPO(REINFORCE):
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)

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

            batch = {
                k: np.repeat(v, self.num_rollouts_per_sample, axis=0)
                for k, v in batch.items()
            }
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
            rollout_res = rollout(
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
            returns = self._compute_returns(
                batch,
                rollout_res,
            )
            total_rollout_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            # Compute log probs
            batch["observations"] = rollout_res.observations
            batch["actions"] = rollout_res.actions
            batch["pred_mask"] = rollout_res.pred_mask
            batch["returns"] = returns
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

            aux[CONST_TRAIN][CONST_SUCCESS_RATE] = np.mean(rollout_res.success).item()
            aux[CONST_TRAIN][CONST_RESPONSE_LENGTH] = np.mean(rollout_res.response_length).item()

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
