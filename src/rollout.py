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
    observations: chex.Array
    actions: chex.Array
    answers: chex.Array
    pointer_correct: chex.Array
    last_prompt_idx: int
    eos: chex.Array
    step_i: int = 0
    deterministic: int = 0
    """
    XXX: Set both to zero if we're using CoT tokens or predicting <EOS>

    XXX: Assume that all possible responses contain tokens with ID up to max_token_id_to_shift
         - You will have to include max_token_id_to_shift + 1 CoT tokens to get the code to run
    """
    correct_aware_shift: int = 0
    max_token_id_to_shift: int = 0


def predict_step(
    step_state: StepState,
):
    step_i = step_state.step_i
    rng, rng_step = jax.random.split(step_state.rng, 2)

    # Autoregressively decode
    module = nnx.merge(step_state.graphdef, step_state.rest, step_state.cache)
    module.eval()
    module.set_attributes(deterministic=True, decode=True)
    logits = module({"sequence": step_state.observations[:, [step_i]],},)
    cache = nnx.state(module, nnx.Cache)

    logits = logits[:, 0]

    # Actual action taken, but output_tokens can be different
    action = jax.lax.cond(
        step_state.deterministic,
        lambda rng_step, logits: jnp.argmax(logits, axis=-1),
        jax.random.categorical,
        rng_step,
        logits,
    )

    is_prompt = step_i < step_state.last_prompt_idx
    pointer_correct = step_state.pointer_correct

    # Shift token by some amount if we know the steps are wrong
    curr_answers = step_state.answers[
        jnp.arange(len(pointer_correct)), pointer_correct
    ]

    reset_pointer = jnp.where(
        step_state.answers[:, 0] == action,
        1,
        0,
    )
    not_prompt_pointer = jax.lax.select(
        curr_answers == action,
        pointer_correct + 1,
        reset_pointer,
    )
    pointer_correct = jax.lax.select(
        is_prompt > 0.0,
        pointer_correct,
        not_prompt_pointer,
    )

    output_tokens = jnp.where(
        jnp.logical_and(
            curr_answers != action,
            action <= step_state.max_token_id_to_shift,
        ),
        action + step_state.correct_aware_shift,
        action,
    )

    # Check for prompt boundary
    output_tokens = jnp.where(
        is_prompt,
        step_state.observations[:, step_i + 1],
        output_tokens,
    )

    # Check if the first EOS has been generated
    eos = jnp.logical_or(
        output_tokens == step_state.eos_token,
        step_state.eos[:, step_i],
    )

    # Update the entries on the i'th step
    observations = step_state.observations.at[:, step_i + 1].set(output_tokens)
    actions = step_state.actions.at[:, step_i + 1].set(action)
    eos = step_state.eos.at[:, step_i + 1].set(eos)

    step_state = StepState(
        graphdef=step_state.graphdef,
        rest=step_state.rest,
        cache=cache,
        eos_token=step_state.eos_token,
        rng=rng,
        observations=observations,
        actions=actions,
        answers=step_state.answers,
        pointer_correct=pointer_correct,
        last_prompt_idx=step_state.last_prompt_idx,
        eos=eos,
        step_i=step_i + 1,
        deterministic=step_state.deterministic,
        correct_aware_shift=step_state.correct_aware_shift,
        max_token_id_to_shift=step_state.max_token_id_to_shift,
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
    deterministic: int = 0,
    correct_aware_shift: int = 0,
    max_token_id_to_shift: int = 0,
):
    questions = batch["sequence"]
    answers = batch["target"]
    pointer_correct = batch["pointer_correct"]
    question_mask = 1 - batch["mask"]
    last_prompt_idx = (
        jnp.sum(question_mask, axis=-1) + pointer_correct - 1
    ).astype(int)
    num_questions, max_step = questions.shape

    step_state = StepState(
        graphdef=graphdef,
        rest=rest,
        cache=cache,
        eos_token=eos_token,
        rng=rng,
        observations=questions,
        actions=jnp.zeros_like(questions, dtype=int),
        answers=answers,
        pointer_correct=pointer_correct,
        last_prompt_idx=last_prompt_idx,
        eos=jnp.zeros((num_questions, max_step)),
        deterministic=deterministic,
        correct_aware_shift=correct_aware_shift,
        max_token_id_to_shift=max_token_id_to_shift,
    )

    step_state = jax.lax.while_loop(
        lambda state: state.step_i < max_step - 1,
        predict_step,
        step_state,
    )

    return (
        step_state.observations,
        step_state.actions,
        step_state.eos,
        question_mask,
        last_prompt_idx,
    )
