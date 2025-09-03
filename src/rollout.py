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

from flax import nnx
from typing import Any, NamedTuple


class StepState(NamedTuple):
    graphdef: Any
    rest: Any
    cache: Any
    eos_token: int
    rng: chex.PRNGKey
    sequence: chex.Array
    is_prompt: chex.Array
    eos: chex.Array
    step_i: int = 0


def predict_step(
    step_state: StepState,
):
    step_i = step_state.step_i
    rng, rng_step = jax.random.split(step_state.rng, 2)

    # Autoregressively decode
    module = nnx.merge(step_state.graphdef, step_state.rest, step_state.cache)
    module.eval()
    module.set_attributes(deterministic=True, decode=True)
    logits = module({"sequence": step_state.sequence[:, [step_i]],},)
    cache = nnx.state(module, nnx.Cache)

    logits = logits[:, 0]
    
    output_tokens = jrandom.categorical(rng_step, logits)

    # Check for prompt boundary
    is_prompt = step_state.is_prompt[:, step_i]

    output_tokens = jnp.where(
        is_prompt,
        step_state.sequence[:, step_i + 1],
        output_tokens,
    )

    # Check if the first EOS has been generated
    eos = jnp.logical_or(
        output_tokens == step_state.eos_token,
        step_state.eos[:, step_i],
    )

    # Update the entries on the i'th step
    sequence = step_state.sequence.at[:, step_i + 1].set(output_tokens)
    eos = step_state.eos.at[:, step_i + 1].set(eos)

    step_state = StepState(
        graphdef=step_state.graphdef,
        rest=step_state.rest,
        cache=cache,
        eos_token=step_state.eos_token,
        rng=rng,
        sequence=sequence,
        is_prompt=step_state.is_prompt,
        eos=eos,
        step_i=step_i + 1,
    )

    return step_state


@nnx.jit
def rollout(
    graphdef: Any,
    cache: Any,
    rest: Any,
    rng: chex.PRNGKey,
    batch: Any,
    eos_token: int,
):
    questions = batch["sequence"]
    mask = batch["mask"]
    num_questions, max_step = questions.shape

    step_state = StepState(
        graphdef=graphdef,
        rest=rest,
        cache=cache,
        eos_token=eos_token,
        rng=rng,
        sequence=questions,
        is_prompt=1 - mask,
        eos=jnp.zeros((num_questions, max_step)),
    )

    step_state = jax.lax.while_loop(
        lambda state: state.step_i < max_step - 1,
        predict_step,
        step_state,
    )

    return (
        step_state.sequence,
        step_state.eos,
        step_state.is_prompt,
    )
