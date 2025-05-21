import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import jax.numpy as jnp

from functools import partial

from src.models.gpt import InContextGPT
from src.models.icrl import (
    XLandADEncoder,
    XLandDPTEncoder,
    ActionTokenLinearPredictor,
    LastActionTokenLinearPredictor,
)

def build_cls(dataset, model_config, rngs, dtype=jnp.float32):
    dependency_cls = {}

    encode_strategy = getattr(
        model_config.model_kwargs,
        "encode_strategy",
        False,
    )
    if encode_strategy:
        if encode_strategy == "xland_ad":
            dependency_cls["encoder_cls"] = partial(
                XLandADEncoder,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                dtype=dtype,
            )
        elif encode_strategy == "xland_dpt":
            dependency_cls["encoder_cls"] = partial(
                XLandDPTEncoder,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
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
        elif predictor_strategy == "last_action_token_linear":
            dependency_cls["predictor_cls"] = partial(
                LastActionTokenLinearPredictor,
                embed_dim=model_config.model_kwargs.embed_dim,
                output_dim=dataset.action_space.n,
                rngs=rngs,
                dtype=dtype,
            )

    return dependency_cls
