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


EOS_TOKEN = 4
class REINFORCE(Learner):
    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)
        self._rng = jrandom.PRNGKey(self._config.seeds.learner_seed)

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
            batch["mask"] = 1 - np.logical_or(eos_mask, is_prompt_mask)
            returns, successes, response_lengths = self.compute_returns(batch)
            batch["returns"] = returns

            batch["entropy_coef"] = getattr(self._config, "entropy", 0.0)
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
    
    def make_validate_step(self):
        if not hasattr(self._config, "validation"):
            print("No validation")
            return

        self.val_dss = {
            validation_config["validation_name"]: get_data_loader(
                parse_dict(validation_config),
                self.data_sharding,
                self.dtype,
            )[0]
            for validation_config in self._config.validation
        }

        def validate_step(epoch: int):
            log = dict()
            curr_rng = jrandom.fold_in(self._rng, epoch)

            for validation_name, val_ds in self.val_dss.items():
                tic = timeit.default_timer()
                batch = next(val_ds)
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
                (responses, eos_mask, is_prompt_mask) = rollout(
                    graphdef,
                    cache,
                    rest,
                    curr_rng,
                    batch,
                    eos_token=EOS_TOKEN,
                )

                batch["sequence"] = responses
                batch["mask"] = 1 - np.logical_or(eos_mask, is_prompt_mask)
                _, successes, response_lengths = self.compute_returns(batch)

                validation_time = timeit.default_timer() - tic

                log[f"time/validation-{validation_name}"] = validation_time
                aux = {
                    CONST_SUCCESS_RATE: np.mean(successes),
                    CONST_RESPONSE_LENGTH: np.mean(response_lengths),
                }
                log.update({
                    f"validation-{validation_name}/{k}": v for k, v in aux.items()
                })

            return log

        self.validation_step = validate_step

    def compute_returns(self, batch):
        # Compute verifiable rewards
        # Assume each token is an action, the state is the sequence up to this point
        # The reward is based on whether there is a regex match with the target

        returns = np.zeros(batch["sequence"].shape)
        response_lengths = np.zeros(batch["sequence"].shape[0])
        successes = np.zeros(batch["sequence"].shape[0])

        # TODO: Entropy regularization objective
        # if getattr(self._config, "regularized_alpha", False):
        #     model = nnx.merge(self.state.graphdef, self.state.params, self.state.rest)
        #     model.set_attributes(deterministic=False, decode=False)
        #     logits = model({
        #         "sequence": batch["sequence"][:, :-1]
        #     })

        #     lprobs = jnp.sum(
        #         nn.one_hot(actions, num_classes=logits.shape[-1]) * logits, axis=-1
        #     ) - nn.logsumexp(logits, axis=-1)

        # TODO: Compute group reward

        for sample_i, (response, target, mask) in enumerate(
            zip(batch["sequence"], batch["target"], batch["mask"])
        ):
            target = "".join(np.array(target[target != EOS_TOKEN]).astype(str)) + "4"

            # XXX: Currently look at the first <EOS>
            # TODO: Maybe we can look at all subsequences between <EQUAL> and <EOS>
            if EOS_TOKEN in response:
                response = "".join(np.array(
                    response[:np.where(response == EOS_TOKEN)[0][0] + 1]
                ).astype(str))
            else:
                response = "".join(np.array(response).astype(str))

            response_length = np.sum(mask)

            success = float(target in (response + "4"))
            reward = success

            reward_type = getattr(self._config, "reward_type", "default")
            if reward_type == "negative_on_failure":
                reward = (-1) ** (1 - success)
            elif reward_type == "negative_dense":
                reward = reward - 1

            response_lengths[sample_i] = response_length
            successes[sample_i] = success
            returns[sample_i][np.where(mask)[0]] = (
                (self._config.gamma ** np.arange(response_length)[::-1]) * reward
            )

            # if getattr(self._config, "regularized_alpha", False):
            #     returns[sample_i] = returns[sample_i] - lprobs * self._config.regularized_alpha
        return returns, successes, response_lengths
