import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


from functools import partial

from src.models.gpt import InContextGPT
from src.models.icrl import XLandSARTSEncoder, ActionTokenLinearPredictor

def build_cls(dataset, model_config, rngs):
    dependency_cls = {}

    encode_strategy = getattr(model_config.model_kwargs, "encode_strategy", False)
    if encode_strategy:
        if encode_strategy == "xland":
            dependency_cls["encoder_cls"] = partial(
                XLandSARTSEncoder,
                embed_dim=model_config.model_kwargs.embed_dim,
                rngs=rngs,
            )
        else:
            raise NotImplementedError

    predictor_strategy = getattr(model_config.model_kwargs, "predictor_strategy", False)
    if predictor_strategy:
        if predictor_strategy == "action_token_linear":
            dependency_cls["predictor_cls"] = partial(
                ActionTokenLinearPredictor,
                embed_dim=model_config.model_kwargs.embed_dim,
                output_dim=dataset.action_space.n,
                rngs=rngs,
            )

    return dependency_cls
