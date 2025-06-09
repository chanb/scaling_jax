import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import jax.numpy as jnp
import numpy as np

from flax import nnx
from flax.training import train_state
from functools import partial
from typing import Any

from src.models.gpt import InContextGPT
from src.models.rnn import InContextGRU
from src.models.supervised import (
    RegressionEmbedders,
)

class TrainState(train_state.TrainState):
    """
    A custom TrainState that includes the model parameters, optimizer state, and additional rest information.
    """
    graphdef: nnx.GraphDef
    rest: Any

def build_cls(dataset, model_config, rngs, dtype=jnp.float32):
    """
    Builds the model and dependency closures based on the dataset and model configuration.
    """
    dependency_cls = {}

    embedder_strategy = getattr(
        model_config.model_kwargs,
        "embedder_strategy",
        False,
    )
    if embedder_strategy:
        if embedder_strategy == "supervised":
            dependency_cls["embedder_cls"] = partial(
                RegressionEmbedders,
                input_dim=int(np.prod(dataset.input_space.shape)),
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                shared_decoding=model_config.model_kwargs.shared_decoding,
                decode=False,
                dtype=dtype,
            )
        else:
            raise NotImplementedError

    return dependency_cls
