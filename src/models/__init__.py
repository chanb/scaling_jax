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

from src.models.attention import quiet_dot_product_attention
from src.models.gpt import InContextGPT
from src.models.next_token import (
    TokenEmbedders,
    VectoredEmbedders,
)
from src.models.rnn import InContextGRU
from src.models.roformer import InContextRoformer
from src.models.supervised import (
    SupervisedEmbedders,
)
from src.models.nope import NoPE
from src.models.sinusoidal_pe import SinusoidalPE

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

    pos_enc_strategy = getattr(
        model_config.model_kwargs,
        "pos_enc_strategy",
        False,
    )

    dependency_cls["pos_enc_cls"] = partial(
        NoPE
    )
    if pos_enc_strategy:
        if pos_enc_strategy == "sinusoidal":
            dependency_cls["pos_enc_cls"] = partial(
                SinusoidalPE,
                embed_dim=model_config.model_kwargs.embed_dim,
                max_len=model_config.model_kwargs.max_decode_len,
                rngs=rngs,
                dtype=dtype,
            )

    embedder_strategy = getattr(
        model_config.model_kwargs,
        "embedder_strategy",
        False,
    )
    if embedder_strategy:
        if embedder_strategy == "regression":
            dependency_cls["embedder_cls"] = partial(
                SupervisedEmbedders,
                input_dim=int(np.prod(dataset.input_space.shape)),
                output_dim=1,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                shared_decoding=model_config.model_kwargs.shared_decoding,
                decode=False,
                dtype=dtype,
            )
        elif embedder_strategy == "classification":
            dependency_cls["embedder_cls"] = partial(
                SupervisedEmbedders,
                input_dim=int(np.prod(dataset.input_space.shape)),
                output_dim=dataset.output_space.n,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                shared_decoding=model_config.model_kwargs.shared_decoding,
                decode=False,
                dtype=dtype,
            )
        elif embedder_strategy == "next_token":
            dependency_cls["embedder_cls"] = partial(
                TokenEmbedders,
                num_input_tokens=dataset.input_space.n,
                num_output_tokens=dataset.output_space.n,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                shared_decoding=model_config.model_kwargs.shared_decoding,
                decode=False,
                dtype=dtype,
            )
        elif embedder_strategy == "next_vector":
            dependency_cls["embedder_cls"] = partial(
                VectoredEmbedders,
                vec_dim=dataset.output_space.shape[0],
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
                shared_decoding=model_config.model_kwargs.shared_decoding,
                decode=False,
                dtype=dtype,
            )
        else:
            raise NotImplementedError

    attention_fn = getattr(
        model_config.model_kwargs,
        "attention_fn",
        False,
    )
    if attention_fn:
        if attention_fn == "quiet_dot_product":
            dependency_cls["attention_fn"] = quiet_dot_product_attention

    return dependency_cls
