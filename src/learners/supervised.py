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
import numpy as np
import timeit


from src.constants import *
from src.dataset import get_data_loader
from src.learners.learner import (
    Learner,
    l2_norm,
    gather_learning_rate,
    initialize_loss_fn,
)
from src.utils import parse_dict


class Supervised(Learner):
    """
    In-context Supervised Learning.
    """

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__(config=config)

    def update(self, epoch: int, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model.

        :param epoch: the epoch
        :type epoch: int
        :return: the update information
        :rtype: Dict[str, Any]

        """
        auxes = []
        total_sample_time = 0
        total_update_time = 0

        for update_i in range(self._num_updates_per_epoch):
            tic = timeit.default_timer()
            batch = self.get_batch()
            total_sample_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            self._state, aux = self.train_step(
                self._state,
                batch,
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS].item()), f"Loss became NaN\naux: {aux}"

            auxes.append(aux)

        auxes = jax.tree_util.tree_map(
            lambda *args: np.mean([np.asarray(el) for el in args]),
            *auxes,
        )

        log = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AGG_LOSS].item(),
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
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
        if not hasattr(self._config, "validate"):
            print("No validation")
            return

        self.val_dss = {
            validation_config["validation_name"]: get_data_loader(
                parse_dict(validation_config),
                self.data_sharding,
                self.dtype,
            )[0]
            for validation_config in self._config.validate
        }

        self._val_loss = initialize_loss_fn(
            self._config.val_objective,
            self._state.graphdef,
            getattr(self._config, "one_hot", False),
        )

        @nnx.jit
        def _compute_metrics(state, batch):
            agg_loss, aux = self._val_loss(
                state.params,
                state.rest,
                batch,
            )
            return agg_loss, aux

        def validate_step(epoch: int):
            log = dict()
            for validation_name, val_ds in self.val_dss.items():
                tic = timeit.default_timer()
                batch = next(val_ds)
                batch = jax.device_put(batch, self.data_sharding)
                agg_loss, aux = _compute_metrics(
                    self._state,
                    batch,
                )
                validation_time = timeit.default_timer() - tic

                aux = jax.tree_util.tree_map(lambda v: np.mean(v).item(), aux)
                log[f"losses/validation-{validation_name}"] = agg_loss.item()
                log[f"time/validation-{validation_name}"] = validation_time
                log.update({
                    f"validation-{validation_name}/{k}": v for k, v in aux[CONST_TRAIN].items()
                })

            return log

        self.validation_step = validate_step
