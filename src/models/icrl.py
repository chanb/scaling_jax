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


class XLandADEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
    ):
        self.decode = decode
        self.embed_dim = embed_dim

        self.entity_emb = nnx.Embed(NUM_TILES + 1, embed_dim // 2, rngs=rngs, dtype=dtype,)
        self.color_emb = nnx.Embed(NUM_COLORS, embed_dim // 2, rngs=rngs, dtype=dtype,)
        self.observation_emb = CNN(
            in_features=embed_dim,
            hidden_features=[32, 16],
            kernel_sizes=[1, 1],
            paddings=[CONST_SAME_PADDING, CONST_SAME_PADDING],
            activation=nnx.relu,
            use_batch_norm=False,
            rngs=rngs,
            dtype=dtype,
        )
        self.observation_projector = nnx.Linear(
            5 * 5 * 16,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        self.action_emb = nnx.Embed(NUM_ACTIONS, embed_dim, rngs=rngs, dtype=dtype,)

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
    

class XLandDPTEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        rngs: nnx.Rngs,
        decode: bool = False,
        dtype=None,
    ):
        self.decode = decode
        self.embed_dim = embed_dim

        self.entity_emb = nnx.Embed(NUM_TILES + 1, embed_dim // 2, rngs=rngs, dtype=dtype,)
        self.color_emb = nnx.Embed(NUM_COLORS, embed_dim // 2, rngs=rngs, dtype=dtype,)
        self.observation_emb = CNN(
            in_features=embed_dim,
            hidden_features=[32, 16],
            kernel_sizes=[1, 1],
            paddings=[CONST_SAME_PADDING, CONST_SAME_PADDING],
            activation=nnx.relu,
            use_batch_norm=False,
            rngs=rngs,
            dtype=dtype,
        )
        self.observation_projector = nnx.Linear(
            5 * 5 * 16,
            embed_dim,
            rngs=rngs,
            dtype=dtype,
        )

        self.action_emb = nnx.Embed(NUM_ACTIONS, embed_dim, rngs=rngs, dtype=dtype,)

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
        if self.decode:
            if "state" in batch:
                obss = batch["state"]
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
                output_sequence = obs_tokens
            elif "action" in batch:
                acts = batch["action"]
                act_tokens = self.action_emb(
                    acts,
                )
                output_sequence = act_tokens
            elif "reward" in batch:
                rews = batch["reward"]
                rew_tokens = self.reward_emb(
                    rews[..., None],
                )
                output_sequence = rew_tokens
            else:
                raise ValueError("Batch must contain 'state', 'action', or 'reward' for decoding.")
        else:
            query_obss = batch["query_state"]
            obss = batch["state"]
            next_obss = batch["next_state"]
            all_obss = jnp.concatenate(
                (
                    obss,
                    next_obss,
                    query_obss,
                ),
                axis=1,
            )
            acts = batch["action"]
            rews = batch["reward"]
            (batch_size, seq_len) = obss.shape[:2]

            entities = self.entity_emb(
                all_obss[..., 0],
            )
            colors = self.color_emb(
                all_obss[..., 1],
            )

            obs_tokens = self.observation_emb(
                jnp.concatenate((entities, colors), axis=-1),
            )
            obs_tokens = obs_tokens.reshape((batch_size, 2 * seq_len + 1, -1))
            obs_tokens = self.observation_projector(obs_tokens)

            act_tokens = self.action_emb(
                acts,
            )

            rew_tokens = self.reward_emb(
                rews[..., None],
            )

            output_sequence = jnp.concatenate((
                obs_tokens[:, :seq_len, None, :],
                act_tokens[:, :, None, :],
                obs_tokens[:, seq_len:2 * seq_len, None, :],
                rew_tokens[:, :, None, :],
            ), axis=2).reshape((
                batch_size,
                4 * seq_len,
                self.embed_dim, # D
            ))

            output_sequence = jnp.concatenate((
                output_sequence,
                obs_tokens[:, -1:, :],
            ), axis=1)

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
            return self.predictor(embed)[:, 1::3]


class LastActionTokenLinearPredictor(nnx.Module):
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
        if self.decode:
            return self.predictor(embed)
        else:
            return self.predictor(embed)[:, -1]
