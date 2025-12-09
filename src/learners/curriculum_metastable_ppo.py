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
from src.dataset import get_data_loader
from src.decoding import make_autoregressive
from src.learners.learner import (
    l2_norm,
    gather_learning_rate,
)
from src.learners.metastable_ppo import MetastablePPO
from src.rollout import rollout
from src.utils import parse_dict


class CurriculumMetastablePPO(MetastablePPO):
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)
        self.boundary_i = 0
        self.num_perfect = 0
        self.update_boundary(replace_ds=False)

    def update_boundary(self, replace_ds=False):
        if self.boundary_i >= len(self.config.dataset_curriculum):
            self.boundary_dataset = None

        if replace_ds:
            self.ds = self._boundary_ds
            self._dataset = self._boundary_dataset

        dataset_config = {
            "dataset_name": self.config.dataset_name,
            "dataset_kwargs": vars(self.config.dataset_kwargs),
            "seeds": vars(self.config.seeds),
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
        }
        dataset_config["dataset_kwargs"].update(
            **self.config.dataset_curriculum[self.boundary_i]
        )
        dataset_config = parse_dict(dataset_config)

        self._boundary_ds, self._boundary_dataset = get_data_loader(
            dataset_config,
            self.data_sharding,
            self.dtype,
        )
        self.boundary_i += 1

    def update(self, epoch: int, *args, **kwargs) -> Dict[str, Any]:
        log = super().update(epoch, *args, **kwargs)

        # Boundary test
        curr_rng = jrandom.fold_in(self._rng, epoch)

        tic = timeit.default_timer()
        batch = self.get_batch()
        batch = next(self._boundary_ds)
        batch = jax.device_put(batch, self.data_sharding)
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
            eos_token=self._boundary_dataset.eos_token_id,
            correct_aware_shift=getattr(self._boundary_dataset, "correctness_aware_tokens_offset", 0),
            max_token_id_to_shift=getattr(self._boundary_dataset, "max_token_id_to_shift", 0),
        )

        # Compute return
        batch["observations"] = observations
        batch["actions"] = actions
        _, successes, _ = self._compute_returns(
            batch,
            last_prompt_idxes,
            is_eval=False,
        )
        mean_success = np.mean(successes)
        if mean_success >= self.config.success_threshold:
            self.num_perfect += 1

        total_boundary_test_time = timeit.default_timer() - tic
        log["time/boundary_time"] = total_boundary_test_time
        log["train/boundary_i"] = self.boundary_i
        log["train/num_perfect"] = self.num_perfect
        log["train/boundary_success_rate"] = mean_success

        if self.num_perfect >= self.config.num_batches_to_consider:
            self.update_boundary(replace_ds=True)
            self.num_perfect = 0

        return log
