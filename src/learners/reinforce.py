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
from src.decoding import make_autoregressive
from src.learners.learner import (
    Learner,
    l2_norm,
    gather_learning_rate,
    EOS_TOKEN,
)
from src.rollout import rollout


class REINFORCE(Learner):
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

            num_rollouts_per_sample = getattr(self._config, "num_rollouts_per_sample", 1)

            batch["sequence"] = np.repeat(batch["sequence"], num_rollouts_per_sample, axis=0)
            batch["mask"] = np.repeat(batch["mask"], num_rollouts_per_sample, axis=0)
            batch["target"] = np.repeat(batch["target"], num_rollouts_per_sample, axis=0)

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
            (responses, eos_mask, is_prompt_mask) = rollout(
                graphdef,
                cache,
                rest,
                curr_rng,
                batch,
                eos_token=EOS_TOKEN,
            )

            # Compute return
            batch["sequence"] = responses
            returns, successes, response_lengths = self._compute_returns(
                batch,
                eos_mask,
                is_prompt_mask,
                is_eval=False,
            )
            batch["returns"] = returns
            total_rollout_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
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
