import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx
from typing import Any
from xminigrid.core.constants import NUM_TILES, NUM_COLORS, NUM_ACTIONS

import jax.numpy as jnp
import numpy as np

from src.constants import *
from src.models.common import CNN, MLP, identity


class XLandSARTSEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim

        self.entity_emb = nnx.Embed(NUM_TILES + 1, embed_dim // 2, rngs=rngs)
        self.color_emb = nnx.Embed(NUM_COLORS, embed_dim // 2, rngs=rngs)
        self.observation_emb = CNN(
            in_features=embed_dim,
            hidden_features=[32, 16],
            kernel_sizes=[1, 1],
            paddings=[CONST_SAME_PADDING, CONST_SAME_PADDING],
            activation=nnx.relu,
            use_batch_norm=False,
            rngs=rngs,
        )
        self.observation_projector = nnx.Linear(
            5 * 5 * 16,
            embed_dim,
            rngs=rngs,
        )

        self.action_emb = nnx.Embed(NUM_ACTIONS, embed_dim, rngs=rngs)

        self.reward_emb = MLP(
            in_dim=1,
            out_dim=embed_dim,
            hidden_layers=[],
            activation=identity,
            rngs=rngs,
            use_layer_norm=False,
            use_batch_norm=False,
            use_bias=True,
        )

        self.terminal_emb = nnx.Embed(
            2,
            embed_dim,
            rngs=rngs
        )

    def __call__(
        self,
        batch: Any,
        **kwargs,
    ):
        obss = batch["state"]
        acts = batch["action"]
        rews = batch["reward"]
        terminals = batch["done"].astype(int)
        (batch_size, seq_len) = obss.shape[:2]

        entities = self.entity_emb(
            obss[..., 0],
        )
        colors = self.color_emb(
            obss[..., 1],
        )

        obs_tokens = self.observation_emb(
            jnp.concatenate((entities, colors), axis=-1),
        )
        obs_tokens = obs_tokens.reshape((batch_size, seq_len, -1))
        obs_tokens = self.observation_projector(obs_tokens)

        act_tokens = self.action_emb(
            acts,
        )

        rew_tokens = self.reward_emb(
            rews[..., None],
        )

        terminal_tokens = self.terminal_emb(
            terminals,
        )

        output_sequence = jnp.concatenate((
            obs_tokens[:, :, None, :],
            act_tokens[:, :, None, :],
            rew_tokens[:, :, None, :],
            terminal_tokens[:, :, None, :],
        ), axis=2).reshape((
            batch_size,
            4 * seq_len,
            self.embed_dim, # D
        ))

        return output_sequence


class ActionTokenLinearPredictor(nnx.Module):
    def __init__(self, embed_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.predictor = nnx.Linear(embed_dim, output_dim, rngs=rngs)

    def __call__(
        self,
        embed,
        *args,
        **kwargs,
    ):
        # Assumes (S, A, R, T, S, A, ...)
        return self.predictor(embed)[:, 1::4]
    