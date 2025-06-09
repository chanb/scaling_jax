import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from typing import Any
from xminigrid.core.constants import NUM_ACTIONS

import jax.numpy as jnp

from src.constants import *
from src.models.common import MLP, identity


class BanditADEncoder(nnx.Module):
    def __init__(
        self,
        num_arms: int,
        embed_dim: int,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
        # TODO: Add dropout and see if that helps at all
    ):
        self.decode = decode
        self.embed_dim = embed_dim
        self.num_arms = num_arms

        self.observation_emb = nnx.Linear(
            1,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        self.action_emb = nnx.Embed(num_arms, embed_dim, rngs=rngs, dtype=dtype,)

        self.reward_emb = MLP(
            in_dim=1,
            out_dim=embed_dim,
            hidden_layers=[],
            activation=identity,
            rngs=rngs,
            dtype=dtype,
            use_layer_norm=False,
            use_batch_norm=False,
            use_bias=True,
        )

    def __call__(
        self,
        batch: Any,
        **kwargs,
    ):
        if "state" in batch:
            obss = batch["state"]
            obs_tokens = self.observation_emb(
                obss,
            )
            output_sequence = obs_tokens
        
        if "action" in batch:
            acts = batch["action"]
            act_tokens = self.action_emb(
                acts,
            )
            output_sequence = act_tokens

        if "reward" in batch:
            rews = batch["reward"]
            rew_tokens = self.reward_emb(
                rews[..., None],
            )
            output_sequence = rew_tokens

        if self.decode:
            return output_sequence
        else:
            (batch_size, seq_len) = obss.shape[:2]
            output_sequence = jnp.concatenate((
                obs_tokens[:, :, None, :],
                act_tokens[:, :, None, :],
                rew_tokens[:, :, None, :],
            ), axis=2).reshape((
                batch_size,
                3 * seq_len,
                self.embed_dim, # D
            ))

        return output_sequence


class ActionTokenLinearPredictor(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
    ):
        self.decode = decode
        self.predictor = nnx.Linear(embed_dim, output_dim, rngs=rngs, dtype=dtype,)

    def __call__(
        self,
        embed,
        *args,
        **kwargs,
    ):
        # Assumes (S, A, R, S, A, ...)
        if self.decode:
            return self.predictor(embed)
        else:
            return self.predictor(embed)[:, ::3]
