import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import jax.numpy as jnp

from flax import nnx
from flax.training import train_state
from functools import partial
from typing import Any

from src.models.gpt import InContextGPT
from src.models.icrl import (
    BanditADEncoder,
    ActionTokenLinearPredictor,
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

    encode_strategy = getattr(
        model_config.model_kwargs,
        "encode_strategy",
        False,
    )
    if encode_strategy:
        if encode_strategy == "bandit_ad":
            dependency_cls["encoder_cls"] = partial(
                BanditADEncoder,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                decode=False,
                dtype=dtype,
            )
        else:
            raise NotImplementedError

    predictor_strategy = getattr(
        model_config.model_kwargs,
        "predictor_strategy",
        False,
    )
    if predictor_strategy:
        if predictor_strategy == "action_token_linear":
            dependency_cls["predictor_cls"] = partial(
                ActionTokenLinearPredictor,
                embed_dim=model_config.model_kwargs.embed_dim,
                output_dim=dataset.action_space.n,
                rngs=rngs,
                dtype=dtype,
            )
        else:
            raise NotImplementedError

    return dependency_cls
