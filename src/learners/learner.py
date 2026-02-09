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
import scipy
import timeit

import src.models as models

from src.constants import *
from src.dataset import get_data_loader
from src.decoding import make_autoregressive
from src.mesh_utils import construct_mesh, construct_sharded_model
from src.rollout import rollout
from src.utils import parse_dict
from src.verifier import make_compute_returns


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

        entropy_coef = loss_config.entropy
        variants = loss_config.mdp_type.split(":")
        if len(variants) == 0 or variants[1] == "default":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / jnp.sum(pred_mask)
        elif variants[1] == "length_bias_fix":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / pred_mask.shape[0]

        if loss_config.mdp_type == "bandit":
            def _compute_loss(lprobs, returns, pred_mask):
                # Objective: log pi(y|s) * R
                lprobs = jnp.sum(lprobs, axis=-1, where=pred_mask)
                return -jnp.mean(
                    lprobs
                    * returns
                )
        elif loss_config.mdp_type.startswith("episodic"):
            def _compute_loss(lprobs, returns, pred_mask):
                # Objective: log pi(a_t|s_t) * G_t
                return -_compute_mean(lprobs * returns * pred_mask, pred_mask)
        else:
            raise NotImplementedError

        def reinforce(params, rest, batch):
            # NOTE: Assume sequence contains both the state and action
            observations = batch["observations"]
            actions = batch["actions"]
            returns = batch["returns"]
            pred_mask = batch["pred_mask"]

            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model({"sequence": observations})

            actions = nn.one_hot(actions, num_classes=logits.shape[-1])
            lprobs = jnp.sum(
                logits, axis=-1, where=actions,
            ) - nn.logsumexp(logits, axis=-1)
            
            probs = nn.softmax(logits, axis=-1)
            entropy = optax.softmax_cross_entropy(logits, probs)
            entropy = _compute_mean(jnp.sum(entropy, where=pred_mask), pred_mask)

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
        variants = loss_config.mdp_type.split(":")
        entropy_coef = loss_config.entropy
        if len(variants) == 0 or variants[1] == "default":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / jnp.sum(pred_mask)
        elif variants[1] == "length_bias_fix":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / pred_mask.shape[0]

        clip_low = getattr(
            loss_config,
            "clip_low",
            getattr(loss_config, "clip_param", 0.2),
        )
        clip_high = getattr(
            loss_config,
            "clip_high",
            getattr(loss_config, "clip_param", 0.2),
        )

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
                    a_min=1 - clip_low,
                    a_max=1 + clip_high,
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
        elif (
            loss_config.mdp_type.startswith("episodic")
            or loss_config.mdp_type.startswith("meta_rl")
            or loss_config.mdp_type.startswith("progress_rl")
            or loss_config.mdp_type.startswith("traj_improvement")
        ):
            def _compute_loss(lprobs, old_lprobs, returns, pred_mask):
                # Objective: log pi(a_t|s_t) * G_t
                is_ratio = jnp.exp(lprobs - old_lprobs)
                # XXX: Deal with inf values
                is_ratio = jax.lax.select(
                    jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
                )

                clipped_is_ratio = jnp.clip(
                    is_ratio,
                    a_min=1 - clip_low,
                    a_max=1 + clip_high,
                )

                surrogate_1 = is_ratio * returns
                surrogate_2 = clipped_is_ratio * returns
                pi_surrogate = jnp.minimum(surrogate_1, surrogate_2)

                is_ratio_max = jnp.max(is_ratio, where=pred_mask, initial=-jnp.inf,)
                is_ratio_min = jnp.min(is_ratio, where=pred_mask, initial=jnp.inf,)
                is_ratio_mean = jnp.nanmean(is_ratio, where=pred_mask)

                return -_compute_mean(pi_surrogate * pred_mask, pred_mask), {
                    "num_clipped": (clipped_is_ratio != is_ratio).sum(),
                    "is_ratio_max": is_ratio_max,
                    "is_ratio_min": is_ratio_min,
                    "is_ratio_mean": is_ratio_mean,
                }
        else:
            raise NotImplementedError

        def ppo(params, rest, batch):
            # NOTE: Assume sequence contains both the state and action
            observations = batch["observations"]
            actions = batch["actions"]
            old_lprobs = batch["old_lprobs"]
            returns = batch["returns"]
            pred_mask = batch["pred_mask"]

            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model({"sequence": observations})

            actions = nn.one_hot(actions, num_classes=logits.shape[-1])
            lprobs = jnp.sum(
                logits, axis=-1, where=actions,
            ) - nn.logsumexp(logits, axis=-1)
            
            probs = nn.softmax(logits, axis=-1)
            entropy = optax.softmax_cross_entropy(logits, probs)
            entropy = _compute_mean(jnp.sum(entropy, where=pred_mask), pred_mask)

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
    elif objective == "ppo_kl":
        variants = loss_config.mdp_type.split(":")
        entropy_coef = loss_config.entropy
        kl_beta = loss_config.kl_beta
        if len(variants) == 0 or variants[1] == "default":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / jnp.sum(pred_mask)
        elif variants[1] == "length_bias_fix":
            def _compute_mean(values, pred_mask):
                return jnp.sum(values) / pred_mask.shape[0]

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

                pi_surrogate = is_ratio * returns

                is_ratio_max = jnp.max(is_ratio)
                is_ratio_min = jnp.min(is_ratio)
                is_ratio_mean = jnp.nanmean(is_ratio)

                return -jnp.mean(pi_surrogate), {
                    "is_ratio_max": is_ratio_max,
                    "is_ratio_min": is_ratio_min,
                    "is_ratio_mean": is_ratio_mean,
                }
        elif (
            loss_config.mdp_type.startswith("episodic")
            or loss_config.mdp_type.startswith("meta_rl")
            or loss_config.mdp_type.startswith("progress_rl")
            or loss_config.mdp_type.startswith("traj_improvement")
        ):
            def _compute_loss(lprobs, old_lprobs, returns, pred_mask):
                # Objective: log pi(a_t|s_t) * G_t
                is_ratio = jnp.exp(lprobs - old_lprobs)
                # XXX: Deal with inf values
                is_ratio = jax.lax.select(
                    jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
                )

                pi_surrogate = is_ratio * returns

                is_ratio_max = jnp.max(is_ratio, where=pred_mask, initial=-jnp.inf,)
                is_ratio_min = jnp.min(is_ratio, where=pred_mask, initial=jnp.inf,)
                is_ratio_mean = jnp.nanmean(is_ratio, where=pred_mask)

                return -_compute_mean(pi_surrogate * pred_mask, pred_mask), {
                    "is_ratio_max": is_ratio_max,
                    "is_ratio_min": is_ratio_min,
                    "is_ratio_mean": is_ratio_mean,
                }
        else:
            raise NotImplementedError

        def ppo(params, rest, batch):
            # NOTE: Assume sequence contains both the state and action
            observations = batch["observations"]
            actions = batch["actions"]
            old_lprobs = batch["old_lprobs"]
            returns = batch["returns"]
            pred_mask = batch["pred_mask"]

            model = nnx.merge(graphdef, params, rest)
            model.set_attributes(deterministic=False, decode=False)
            logits = model({"sequence": observations})

            actions = nn.one_hot(actions, num_classes=logits.shape[-1])
            lprobs = jnp.sum(
                logits, axis=-1, where=actions,
            ) - nn.logsumexp(logits, axis=-1)
            
            probs = nn.softmax(logits, axis=-1)
            entropy = optax.softmax_cross_entropy(logits, probs)
            entropy = _compute_mean(jnp.sum(entropy, where=pred_mask), pred_mask)

            log_ratio = old_lprobs - lprobs
            kl_reg = _compute_mean(
                jnp.sum((jnp.exp(log_ratio) - 1) - log_ratio, where=pred_mask),
                pred_mask,
            )

            entropy_loss = -entropy
            ppo_loss, aux = _compute_loss(lprobs, old_lprobs, returns, pred_mask)

            return ppo_loss + entropy_coef * entropy_loss + kl_beta * kl_reg, {
                CONST_TRAIN: {
                    "entropy": entropy,
                    "pi_loss": ppo_loss,
                    "kl_reg": kl_reg,
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

        self._rng = jrandom.PRNGKey(self._config.seeds.learner_seed)
        self._initialize_model_and_opt(self.dtype)

        self._loss = initialize_loss_fn(
            self._config.train_loss_config,
            self._state.graphdef,
            getattr(self._config, "one_hot", False),
        )
        self._compute_returns = make_compute_returns(
            self._config,
            self._dataset.eos_token_id,
            self._dataset.reset_token_id,
            self._dataset.token_map,
        )
        self.train_step = nnx.jit(self.make_train_step())
        self.make_validate_step()

    def close(self):
        del self.ds

    @property
    def config(self):
        return self._config

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

        model_kwargs = vars(self._config.model_config.model_kwargs)
        model_kwargs.update(dependency_cls)
        self._state, self._state_sharding = construct_sharded_model(
            self.data_mesh,
            model_cls,
            dict(
                **model_kwargs,
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

    def make_validate_step(self):
        if not hasattr(self._config, "validation"):
            print("No validation")
            return

        self.val_dss = {
            validation_config["validation_name"]: (
                get_data_loader(
                    parse_dict(validation_config),
                    self.data_sharding,
                    self.dtype,
                )[0],
                validation_config.get("num_rollouts_per_sample", 1),
                validation_config["attempt_length"],
            )
            for validation_config in self._config.validation
        }

        def validate_step(epoch: int):
            log = dict()
            curr_rng = jrandom.fold_in(self._rng, epoch)

            for validation_name, (val_ds, num_rollouts_per_sample, attempt_length) in self.val_dss.items():
                tic = timeit.default_timer()
                batch = next(val_ds)
                batch = {
                    k: np.repeat(v, num_rollouts_per_sample, axis=0)
                    for k, v in batch.items()
                }
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
                rollout_res = rollout(
                    graphdef,
                    cache,
                    rest,
                    curr_rng,
                    batch,
                    eos_token=self._dataset.eos_token_id,
                    attempt_length=attempt_length,
                    deterministic=int(num_rollouts_per_sample == 1),
                    correct_aware_shift=getattr(self._dataset, "correctness_aware_tokens_offset", 0),
                    max_token_id_to_shift=getattr(self._dataset, "max_token_id_to_shift", 0),
                )

                successes = rollout_res.success
                response_lengths = rollout_res.response_length

                validation_time = timeit.default_timer() - tic

                log[f"time/validation-{validation_name}"] = validation_time
                aux = {
                    CONST_SUCCESS_RATE: np.mean(successes).item(),
                    CONST_RESPONSE_LENGTH: np.mean(response_lengths).item(),
                }
                if num_rollouts_per_sample > 1:
                    success_per_sample = np.sum(successes.reshape((-1, num_rollouts_per_sample)), axis=-1)
                    pass_k = np.mean(
                        1 - scipy.special.comb(
                            num_rollouts_per_sample - success_per_sample,
                            np.full_like(success_per_sample, fill_value=num_rollouts_per_sample // 2),
                            exact=False,
                        ) / scipy.special.comb(num_rollouts_per_sample, num_rollouts_per_sample // 2, exact=False), axis=0
                    )
                    aux["pass@{}".format(num_rollouts_per_sample // 2)] = pass_k

                log.update({
                    f"validation-{validation_name}/{k}": v for k, v in aux.items()
                })

            return log

        self.validation_step = validate_step
