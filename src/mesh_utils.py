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
from typing import Any, Dict, Sequence, Callable

import jax

from src.optimizer import get_optimizer


def construct_mesh(mesh_config: SimpleNamespace):
    num_devices = len(jax.devices())

    mesh_keys = ("data", "fsdp", "tensor",)
    mesh_vals = np.array(
        [getattr(mesh_config, mesh_key) for mesh_key in mesh_keys]
    )
    unspecified_idx = np.where(mesh_vals == -1)[0]
    assert len(unspecified_idx) <= 1

    if len(unspecified_idx) == 1:
        rest_prod = int(np.prod(mesh_vals) * -1)
        assert num_devices % rest_prod == 0
        mesh_vals[unspecified_idx] = num_devices // rest_prod

    print("Mesh shape: {}".format(mesh_vals))
    return Mesh(
        create_device_mesh(mesh_vals),
        mesh_keys,
    )


def construct_sharded_model(
    mesh: Mesh,
    model_cls: Callable,
    model_kwargs: Dict,
    opt_config: SimpleNamespace
):
    def _to_array(x):
        if not isinstance(x, jax.Array):
            x = jnp.asarray(x)
        return x

    @nnx.jit
    def create_sharded_model():
        model = model_cls(
            **model_kwargs,
        )

        opt = get_optimizer(opt_config)

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
    
    with mesh:
        state = create_sharded_model()
        state_sharding = nnx.get_named_sharding(state, mesh)
        return state, state_sharding
