import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flax import nnx
from flax.training import train_state
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import timeit

import src.models as models

from src.constants import *
from src.dataset import get_data_loader
from src.optimizer import get_optimizer


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
    for opt_state in opt_state_list:
        hyperparams = getattr(opt_state, CONST_HYPERPARAMS, {})
        if CONST_LEARNING_RATE in hyperparams:
            aux[f"{CONST_LEARNING_RATE}/{model_name}"] = hyperparams[
                CONST_LEARNING_RATE
            ].item()


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    rest: Any


class ICRL:
    """
    In-context Reinforcement Learning.
    """

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        self._config = config
        self._num_updates_per_epoch = config.num_updates_per_epoch
        self._learner_key = jrandom.PRNGKey(config.seeds.learner_seed)
        self.construct_mesh()

        self.data_sharding = NamedSharding(self.data_mesh, P("data"))

        dtype = jnp.float32
        if self._config.half_precision:
            dtype = jnp.bfloat16

        self.ds, self._dataset = get_data_loader(
            config,
            self.data_sharding,
            dtype,
        )

        self._initialize_model_and_opt(dtype)
        self._initialize_losses()
        self.train_step = nnx.jit(self.make_train_step())

    def construct_mesh(self):
        self.num_devices = len(jax.devices())

        mesh_keys = ("data", "fsdp", "tensor",)
        mesh_vals = np.array(
            [getattr(self._config.mesh, mesh_key) for mesh_key in mesh_keys]
        )
        unspecified_idx = np.where(mesh_vals == -1)[0]
        assert len(unspecified_idx) <= 1

        if len(unspecified_idx) == 1:
            rest_prod = int(np.prod(mesh_vals) * -1)
            assert self.num_devices % rest_prod == 0
            mesh_vals[unspecified_idx] = self.num_devices // rest_prod

        print("Mesh shape: {}".format(mesh_vals))
        self.data_mesh = Mesh(
            create_device_mesh(mesh_vals),
            mesh_keys,
        )

    def close(self):
        del self.ds

    @property
    def model(self):
        """
        Model
        """
        return self._model

    @property
    def state(self):
        """
        Model states
        """
        self._state

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

        def _to_array(x):
            if not isinstance(x, jax.Array):
                x = jnp.asarray(x)
            return x

        @nnx.jit
        def create_sharded_model():
            model = model_cls(
                **vars(self._config.model_config.model_kwargs),
                **dependency_cls,
                rngs=rngs,
                dtype=dtype,
            )

            opt = get_optimizer(self._config.optimizer_config)

            graphdef, params, rest = nnx.split(model, nnx.Param, ...)

            state = TrainState.create(
                apply_fn=graphdef.apply,
                params=params,
                tx=opt,
                graphdef=graphdef,
                rest=rest,
            )
            state = jax.tree.map(_to_array, state)
            state_spec = nnx.get_partition_spec(state)
            state = jax.lax.with_sharding_constraint(state, state_spec)
            return state

        with self.data_mesh:
            state = create_sharded_model()
            state_sharding = nnx.get_named_sharding(state, self.data_mesh)
            self._state = state
            self._state_sharding = state_sharding

    def _initialize_losses(self):
        if self._config.objective == "mle":
            def cross_entropy(params, rest, batch):
                targets = batch["target"]
                model = nnx.merge(self._state.graphdef, params, rest)
                model.set_attributes(deterministic=False, decode=False)
                logits = model(batch)
                loss = jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, targets)
                )
                acc = jnp.mean(
                    jnp.argmax(logits, axis=-1) == targets
                )

                return loss, {
                    CONST_ACCURACY: acc,
                }

            self._loss = cross_entropy
        else:
            raise NotImplementedError

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
            batch = next(self.ds)
            batch = jax.device_put(batch, self.data_sharding)
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
            f"losses/{CONST_ACCURACY}": auxes[CONST_ACCURACY].item(),
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"{CONST_GRAD_NORM}/model": auxes[CONST_GRAD_NORM][CONST_MODEL].item(),
            f"{CONST_PARAM_NORM}/model": l2_norm(self._state.params).item(),
        }

        if isinstance(self._state.opt_state, dict):
            for model_name, optimizer in self._state.opt_state:
                gather_learning_rate(aux, model_name, optimizer)
        else:
            gather_learning_rate(aux, CONST_MODEL, self._state.opt_state)
        return log
