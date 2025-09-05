import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import chex
import dill
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

import src.models as models

from src.constants import *
from src.dataset import get_data_loader
from src.mesh_utils import construct_mesh, construct_sharded_model


def l2_norm(params: chex.PyTreeDef) -> chex.Array:
    """
    Computes the L2 norm of a complete PyTree.

    :param params: the pytree object with scalars
    :type params: PyTreeDef
    :return: L2 norm of the complete PyTree
    :rtype: chex.Array

    """
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))


def gather_learning_rate(
    aux: Dict,
    model_name: str,
    opt_state_list: Sequence[Any],
):
    """
    Gathers the learning rate from the optimizer state and adds it to the aux dictionary.
    """
    for opt_state in opt_state_list:
        hyperparams = getattr(opt_state, CONST_HYPERPARAMS, {})
        if CONST_LEARNING_RATE in hyperparams:
            aux[f"{CONST_LEARNING_RATE}/{model_name}"] = hyperparams[
                CONST_LEARNING_RATE
            ].item()


def initialize_loss_fn(loss_config, graphdef, one_hot=False):
    objective = loss_config.objective
    if objective == "ce":
        if one_hot:
            def compute_target(targets):
                return jnp.argmax(targets, axis=-1)
        else:
            def compute_target(targets):
                return targets

        def cross_entropy(params, rest, batch):
            targets = compute_target(batch["target"])
            mask = batch["mask"]
            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model(batch)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            acts_taken = jnp.argmax(logits, axis=-1)
            acc = acts_taken == targets

            return jnp.sum(loss * mask) / jnp.sum(mask), {
                CONST_TRAIN: {
                    **{
                        f"{CONST_ACCURACY}-context_{context_i}": jnp.mean(acc[:, context_i])
                        for context_i in range(acc.shape[1])
                    },
                    **{
                        f"{CONST_LOSS}-context_{context_i}": jnp.mean(loss[:, context_i])
                        for context_i in range(loss.shape[1])
                    },
                },
                CONST_HIST: {
                    CONST_ACT_TAKEN: acts_taken,
                    CONST_ACT_TARGET: targets,
                },
            }

        return cross_entropy
    elif objective == "mse":
        def mse(params, rest, batch):
            targets = batch["target"]
            mask = batch["mask"]
            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            preds = model(batch)

            loss = optax.squared_error(preds, targets)

            return jnp.sum(loss * mask) / jnp.sum(mask), {
                CONST_TRAIN: {
                    **{
                        f"{CONST_LOSS}-context_{context_i}": jnp.mean(loss[:, context_i])
                    for context_i in range(loss.shape[1])
                    }
                },
                CONST_HIST: {},
            }

        return mse
    elif objective == "contrastive":
        def contrastive(params, rest, batch):
            targets = batch["target"][:, [-1]]
            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            preds = model(batch)[:, [-1]]
            contextless_preds = model({
                "sequence": batch["sequence"][:, [-1]]
            })

            pos_loss = optax.squared_error(preds, targets)
            neg_loss = -jnp.clip(
                optax.squared_error(contextless_preds, targets),
                a_min=0.0,
                a_max=1.0,
            )

            return jnp.mean(pos_loss + neg_loss), {
                CONST_TRAIN: {
                    "pos_loss": jnp.mean(pos_loss),
                    "neg_loss": jnp.mean(neg_loss),
                },
                CONST_HIST: {},
            }

        return contrastive
    elif objective == "reinforce":
        # TODO: Add KL regularizer to reference model

        if loss_config.mdp_type == "bandit":
            def _compute_loss(lprobs, returns, pred_mask):
                # Objective: log pi(y|s) * R
                lprobs = jnp.sum(lprobs, axis=-1, where=pred_mask)
                return -jnp.mean(
                    lprobs
                    * returns
                )
        elif loss_config.mdp_type == "episodic":
            def _compute_loss(lprobs, returns, pred_mask):
                # Objective: log pi(a_t|s_t) * G_t
                returns = returns[:, :-1]
                return -jnp.sum(lprobs * returns * pred_mask) / jnp.sum(pred_mask)
        else:
            raise NotImplementedError

        def reinforce(params, rest, batch):
            # NOTE: Assume sequence contains both the state and action
            observations = batch["sequence"][:, :-1]
            actions = batch["sequence"][:, 1:]
            # returns = batch["returns"][:, :-1]
            returns = batch["returns"]
            pred_mask = batch["pred_mask"][:, :-1]
            # first_eos_mask = batch["first_eos_mask"][:, :-1]
            entropy_coef = batch["entropy_coef"]

            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model({"sequence": observations})

            actions = nn.one_hot(actions, num_classes=logits.shape[-1])
            lprobs = jnp.sum(
                logits, axis=-1, where=actions,
            ) - nn.logsumexp(logits, axis=-1)
            
            probs = nn.softmax(logits, axis=-1)
            entropy = optax.softmax_cross_entropy(logits, probs)
            entropy = jnp.mean(entropy, where=pred_mask)

            reinforce_loss = _compute_loss(lprobs, returns, pred_mask)

            entropy_loss = -entropy

            return reinforce_loss + entropy_coef * entropy_loss, {
                CONST_TRAIN: {
                    "entropy": entropy,
                    "pi_loss": reinforce_loss,
                },
                CONST_HIST: {},
            }
        return reinforce
    elif objective == "ppo":

        if loss_config.mdp_type == "bandit":
            def _compute_loss(lprobs, old_lprobs, returns, pred_mask):
                # Objective: log pi(y|s) * R
                lprobs = jnp.sum(lprobs, axis=-1, where=pred_mask)
                old_lprobs = jnp.sum(old_lprobs, axis=-1, where=pred_mask)

                is_ratio = jnp.exp(lprobs - old_lprobs)
                # XXX: Deal with inf values
                is_ratio = jax.lax.select(
                    jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
                )

                clipped_is_ratio = jnp.clip(
                    is_ratio,
                    a_min=1 - loss_config.clip_param,
                    a_max=1 + loss_config.clip_param,
                )

                surrogate_1 = is_ratio * returns
                surrogate_2 = clipped_is_ratio * returns
                pi_surrogate = jnp.minimum(surrogate_1, surrogate_2)

                is_ratio_max = jnp.max(is_ratio)
                is_ratio_min = jnp.min(is_ratio)
                is_ratio_mean = jnp.nanmean(is_ratio)

                return -jnp.mean(pi_surrogate), {
                    "num_clipped": (clipped_is_ratio != is_ratio).sum(),
                    "is_ratio_max": is_ratio_max,
                    "is_ratio_min": is_ratio_min,
                    "is_ratio_mean": is_ratio_mean,
                }
        elif loss_config.mdp_type == "episodic":
            def _compute_loss(lprobs, old_lprobs, returns, pred_mask):
                # Objective: log pi(a_t|s_t) * G_t
                returns = returns[:, :-1]
                is_ratio = jnp.exp(lprobs - old_lprobs)
                # XXX: Deal with inf values
                is_ratio = jax.lax.select(
                    jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
                )

                clipped_is_ratio = jnp.clip(
                    is_ratio,
                    a_min=1 - loss_config.clip_param,
                    a_max=1 + loss_config.clip_param,
                )

                surrogate_1 = is_ratio * returns
                surrogate_2 = clipped_is_ratio * returns
                pi_surrogate = jnp.minimum(surrogate_1, surrogate_2)

                is_ratio_max = jnp.max(is_ratio, where=pred_mask)
                is_ratio_min = jnp.min(is_ratio, where=pred_mask)
                is_ratio_mean = jnp.nanmean(is_ratio, where=pred_mask)

                return -jnp.sum(pi_surrogate * pred_mask) / jnp.sum(pred_mask), {
                    "num_clipped": (clipped_is_ratio != is_ratio).sum(),
                    "is_ratio_max": is_ratio_max,
                    "is_ratio_min": is_ratio_min,
                    "is_ratio_mean": is_ratio_mean,
                }
        else:
            raise NotImplementedError

        def ppo(params, rest, batch):
            # NOTE: Assume sequence contains both the state and action
            observations = batch["sequence"][:, :-1]
            actions = batch["sequence"][:, 1:]
            old_lprobs = batch["old_lprobs"]
            # returns = batch["returns"][:, :-1]
            returns = batch["returns"]
            pred_mask = batch["pred_mask"][:, :-1]
            # first_eos_mask = batch["first_eos_mask"][:, :-1]
            entropy_coef = batch["entropy_coef"]

            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model({"sequence": observations})

            actions = nn.one_hot(actions, num_classes=logits.shape[-1])
            lprobs = jnp.sum(
                logits, axis=-1, where=actions,
            ) - nn.logsumexp(logits, axis=-1)
            
            probs = nn.softmax(logits, axis=-1)
            entropy = optax.softmax_cross_entropy(logits, probs)
            entropy = jnp.mean(entropy, where=pred_mask)

            entropy_loss = -entropy
            ppo_loss, aux = _compute_loss(lprobs, old_lprobs, returns, pred_mask)

            return ppo_loss + entropy_coef * entropy_loss, {
                CONST_TRAIN: {
                    "entropy": entropy,
                    "pi_loss": ppo_loss,
                    **{k: v for k, v in aux.items()},
                },
                CONST_HIST: {},
            }
        return ppo
    else:
        raise NotImplementedError


class Learner:
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        self._config = config
        self._num_updates_per_epoch = config.num_updates_per_epoch
        self._learner_key = jrandom.PRNGKey(config.seeds.learner_seed)
        self.data_mesh = construct_mesh(config.mesh)

        self.data_sharding = NamedSharding(self.data_mesh, P("data"))

        self.dtype = jnp.float32
        if self._config.half_precision:
            self.dtype = jnp.bfloat16

        self.ds, self._dataset = get_data_loader(
            config,
            self.data_sharding,
            self.dtype,
        )

        self._initialize_model_and_opt(self.dtype)

        self._loss = initialize_loss_fn(
            self._config.train_loss_config,
            self._state.graphdef,
            getattr(self._config, "one_hot", False),
        )
        self.train_step = nnx.jit(self.make_train_step())
        self.make_validate_step()

    def close(self):
        del self.ds

    @property
    def state(self):
        """
        Model states
        """
        return self._state

    def _initialize_model_and_opt(self, dtype):
        """
        Construct the model and the optimizer.
        """
        
        rngs = nnx.Rngs(self._config.seeds.learner_seed)

        model_cls = getattr(
            models,
            self._config.model_config.architecture,
        )
        dependency_cls = models.build_cls(
            self._dataset,
            self._config.model_config,
            rngs=rngs,
            dtype=dtype,
        )

        self._state, self._state_sharding = construct_sharded_model(
            self.data_mesh,
            model_cls,
            dict(
                **vars(self._config.model_config.model_kwargs),
                **dependency_cls,
                rngs=rngs,
                dtype=dtype,
            ),
            self._config.optimizer_config,
        )

        if hasattr(self._config, "load_checkpoint"):
            load_path, checkpoint_i = self._config.load_checkpoint.split(":")
            all_steps = sorted(os.listdir(os.path.join(load_path, "models")))
            if checkpoint_i == "latest":
                step = all_steps[-1]
            else:
                step = np.argmin(
                    np.abs(
                        np.array([int(step.split(".")[0]) for step in all_steps])
                        - int(checkpoint_i)
                    )
                )
                step = all_steps[step]

            print("Loading checkpoint {} at step {}".format(load_path, step))
            self._state = dill.load(open(os.path.join(load_path, "models", step), "rb"))

    def get_batch(self):
        batch = next(self.ds)
        batch = jax.device_put(batch, self.data_sharding)
        return batch

    def make_train_step(self):
        """
        Makes the training step for model update.
        """

        def _train_step(
            state,
            batch,
            *args,
            **kwargs,
        ) -> Any:
            grad_fn = jax.value_and_grad(self._loss, has_aux=True)
            (agg_loss, aux), grads = grad_fn(
                state.params,
                state.rest,
                batch,
            )

            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {CONST_MODEL: l2_norm(grads)}

            new_state = state.apply_gradients(grads=grads)

            return new_state, aux

        return _train_step
