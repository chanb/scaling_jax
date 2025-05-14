from flax import nnx
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import jax
import jax.numpy as jnp
import optax


# Create mesh
num_devices = len(jax.devices())
data_mesh = Mesh(
    create_device_mesh((num_devices,)),
    ('data',),
)


class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.w1 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (din, dmid)),
        )

        self.b1 = nnx.Param(
            jnp.zeros((dmid,)),
        )

        self.w2 = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (dmid, dout)),
        )

    def __call__(self, x: jax.Array):
        return nnx.relu(x @ self.w1 + self.b1) @ self.w2


in_dim = 2
out_dim = 1

@nnx.jit
def create_sharded_model():
    model = MLP(in_dim, 64, out_dim, rngs=nnx.Rngs(0)) # Unsharded at this moment.
    state = nnx.state(model)                   # The model's state, a pure pytree.
    pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)           # The model is sharded now!
    return model

with data_mesh:
    sharded_model = create_sharded_model()

optimizer = nnx.Optimizer(
    sharded_model,
    optax.inject_hyperparams(optax.adam)(1e-3),
)

import ipdb
ipdb.set_trace()

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model: MLP):
        y_pred = model(x)

        print(y_pred.shape, x.shape, y.shape)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss

data_sharding = NamedSharding(data_mesh, P('data'))

batch_size_per_device = 2
num_samples = num_devices * batch_size_per_device
seq_len = 2
input = jax.device_put(jax.random.normal(jax.random.key(1), (num_samples, seq_len, in_dim)), data_sharding)
label = jax.device_put(jax.random.normal(jax.random.key(2), (num_samples, seq_len, out_dim)), data_sharding)

for i in range(100000):
    loss = train_step(sharded_model, optimizer, input, label)
    print(loss)    # Model (over-)fitting to the labels quickly.
