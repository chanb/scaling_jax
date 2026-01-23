import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flax import nnx, struct
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple

import chex
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


class Minibatch(NamedTuple):
    context: chex.Array
    target: chex.Array
    question_len: chex.Array
    solution_len: chex.Array
    pointer_correct: chex.Array


class ContextBuffer(struct.PyTreeNode):
    size: int = struct.field(pytree_node=False)
    data: chex.ArrayTree
    index: int
    full: bool

    @classmethod
    def empty(cls, size: int, max_seq_len: int) -> "ContextBuffer":
        data = Minibatch(
            context=jnp.empty((size, max_seq_len + 1), dtype=int),
            target=jnp.empty((size, max_seq_len + 1), dtype=int),
            pointer_correct=jnp.empty((size, max_seq_len + 1), dtype=int),
            question_len=jnp.empty((size,), dtype=int),
            solution_len=jnp.empty((size,), dtype=int),
        )
        return cls(size=size, data=data, index=0, full=False)

    @property
    def num_entries(self):
        return jnp.where(self.full, self.size, self.index)

    @jax.jit
    def append(self, a: chex.ArrayTree) -> "ContextBuffer":
        data = jax.tree.map(lambda arr, a_: arr.at[self.index].set(a_), self.data, a)
        next_index = (self.index + 1) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)

    @jax.jit
    def extend(self, batch: chex.ArrayTree) -> "ContextBuffer":
        batch_flat, _ = jax.tree.flatten(batch)
        batch_size = batch_flat[0].shape[0]

        idx = self.index + jnp.arange(batch_size)
        idx = idx % self.size
        data = jax.tree.map(lambda arr, b: arr.at[idx].set(b), self.data, batch)

        next_index = (self.index + batch_size) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)

    def __getattr__(self, name):
        if name in self.data._fields:
            return getattr(self.data, name)

    @partial(jax.jit, static_argnames=("num"))
    def sample(self, num: int, rng: chex.PRNGKey) -> Minibatch:
        minibatch_index = jax.random.randint(rng, (num,), 0, self.num_entries)
        return jax.tree.map(lambda arr: arr[minibatch_index], self.data)


class OffPolicyContextPPO(REINFORCE):
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)
        self.buffer = ContextBuffer.empty(config.buffer_size, config.max_seq_len)

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

            if epoch > 0:
                minibatch = self.buffer.sample(self._config.batch_size, curr_rng)
                min_idx = jnp.max(minibatch.question_len) + 1
                max_idx = jnp.min(jnp.argmax(minibatch.context == self._dataset.eos_token_id))
                random_idx = jrandom.randint(curr_rng, shape=(), minval=min_idx, maxval=max_idx)

                batch["sequence"] = jnp.concatenate((
                    batch["sequence"],
                    minibatch.context.at[:, random_idx:].set(self._dataset.eos_token_id),
                ), axis=0)
                batch["target"] = jnp.concatenate((
                    batch["target"], minibatch.target,
                ), axis=0)
                batch["mask"] = jnp.concatenate((
                    batch["mask"],
                    jnp.cumsum(
                        jnp.zeros_like(batch["mask"], dtype=int).at[:, random_idx - 1].set(1),
                        axis=-1,
                        dtype=int,
                    ),
                ), axis=0)
                batch["question_len"] = jnp.concatenate((
                    batch["question_len"], minibatch.question_len,
                ), axis=0)
                batch["solution_len"] = jnp.concatenate((
                    batch["solution_len"], minibatch.solution_len,
                ), axis=0)
                batch["pointer_correct"] = jnp.concatenate((
                    batch["pointer_correct"], minibatch.pointer_correct[:, random_idx - 1],
                ), axis=0)

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

            # TODO: Only store new samples from dataset.
            # TODO: Maybe only store when success rate is poor.
            self.buffer = self.buffer.extend(Minibatch(
                context=jnp.hstack((
                    rollout_res.observations.at[jnp.where(
                        jnp.clip(
                            rollout_res.pred_mask + (1 - batch["mask"][:, :-1]),
                            a_min=0,
                            a_max=1,
                        ) == 0
                    )].set(self._dataset.eos_token_id),
                    jnp.full(
                        (len(rollout_res.observations), 1),
                        fill_value=self._dataset.eos_token_id,
                        dtype=int,
                    ),
                )),
                pointer_correct=jnp.hstack((
                    rollout_res.pointer_correct,
                    jnp.full(
                        (len(rollout_res.pointer_correct), 1),
                        fill_value=-1,
                        dtype=int,
                    ),
                )),
                target=jnp.hstack((
                    batch["target"][:, :-1],
                    jnp.full(
                        (len(rollout_res.pointer_correct), 1),
                        fill_value=self._dataset.eos_token_id,
                        dtype=int,
                    ),
                )),
                question_len=batch["question_len"],
                solution_len=batch["solution_len"],
            ))
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
