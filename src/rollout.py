import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom

from functools import partial
from typing import Any, NamedTuple


class StepState(NamedTuple):
    cache: Any
    rng: chex.PRNGKey
    sequence: chex.Array
    take_prediction: chex.Array
    eos: chex.Array
    step_i: int = 0


# TODO: Rollout
"""
This should be autoregressive, from step 0
1. At step i, we replace the i'th index if the entry is not part of the question
2. Otherwise, continue generating

This can be done using mask cond
"""

def predict_step(
    step_state: StepState,
    decode: callable,
    eos_token: int,
):
    step_i = step_state.step_i
    rng, rng_step = jax.random.split(step_state.rng, 2)

    # Autoregressively decode
    logits, cache = decode(
        {"sequence": step_state.sequence[:, [step_i]],},
        step_state.cache,
    )
    logits = logits[:, 0]
    
    output_tokens = jrandom.categorical(rng_step, logits)

    # Check for prompt boundary
    take_prediction = step_state.take_prediction[:, step_i]

    output_tokens = jnp.where(
        take_prediction,
        step_state.sequence[:, step_i + 1],
        output_tokens
    )

    # Check if the first EOS has been generated
    eos = jnp.logical_or(
        output_tokens == eos_token,
        step_state.eos[:, step_i],
    )

    # Update the entries on the i'th step
    sequence = step_state.sequence.at[:, step_i + 1].set(output_tokens)
    eos = step_state.eos.at[:, step_i + 1].set(eos)

    step_state = StepState(
        cache=cache,
        rng=rng,
        sequence=sequence,
        take_prediction=step_state.take_prediction,
        eos=eos,
        step_i=step_i + 1,
    )

    return step_state


def rollout(
    rng: chex.PRNGKey,
    batch: Any,
    decode: callable,
    init_cache: callable,
    eos_token: int,
):
    questions = batch["sequence"]
    mask = batch["mask"]
    num_questions, max_step = questions.shape

    cache = init_cache()
    step_state = StepState(
        cache=cache,
        rng=rng,
        sequence=questions,
        take_prediction=1 - mask,
        eos=jnp.zeros((num_questions, max_step)),
    )

    _predict_step = jax.jit(partial(predict_step, decode=decode, eos_token=eos_token))
    step_state = jax.lax.while_loop(
        lambda state: state.step_i < max_step - 1,
        _predict_step,
        step_state,
    )

    return (
        step_state.sequence,
        step_state.eos,
        step_state.take_prediction,
    )
