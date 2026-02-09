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
from functools import partial
from typing import Any, NamedTuple


class StepState(NamedTuple):
    graphdef: Any
    rest: Any
    cache: Any
    eos_token: int
    rng: chex.PRNGKey
    observations: chex.Array
    actions: chex.Array
    solution: chex.Array
    solution_len: chex.Array
    solution_found: chex.Array
    pointer_correct: chex.Array
    last_prompt_idx: int
    eos: chex.Array
    attempt_step_i: chex.Array
    attempt_length: int
    pred_mask: chex.Array
    step_i: int = 0
    deterministic: int = 0
    """
    XXX: Set both to zero if we're using CoT tokens or predicting <EOS>

    XXX: Assume that all possible responses contain tokens with ID up to max_token_id_to_shift
         - You will have to include max_token_id_to_shift + 1 CoT tokens to get the code to run
    """
    correct_aware_shift: int = 0
    max_token_id_to_shift: int = 0


class RolloutResult(NamedTuple):
    observations: chex.Array
    actions: chex.Array
    solution_found: chex.Array
    success: chex.Array
    response_length: chex.Array
    pointer_correct: chex.Array
    pred_mask: chex.Array
    last_prompt_idx: int
    eos: chex.Array


def predict_step(
    step_state: StepState,
):
    step_i = step_state.step_i
    attempt_step_i = step_state.attempt_step_i
    rng, rng_step = jax.random.split(step_state.rng, 2)
    is_prompt = step_i < step_state.last_prompt_idx
    pred_mask = step_state.pred_mask[:, step_i]
    pointer_correct = step_state.pointer_correct[:, step_i]
    solution_found = step_state.solution_found[:, step_i]

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

    # Shift token by some amount if we know the steps are wrong
    curr_soln_token = step_state.solution[
        jnp.arange(len(pointer_correct)), pointer_correct
    ]

    attempt_step_i = jnp.where(
        step_state.observations[:, step_i + 1] == step_state.solution[:, 0],
        0,
        (attempt_step_i + 1) % step_state.attempt_length,
    )

    # jax.debug.print("{x}, {y}", x=attempt_step_i, y=step_state.observations[:, step_i])

    pred_mask = step_state.pred_mask.at[:, step_i].set(
        jnp.where(
            jnp.logical_or(
                is_prompt,
                attempt_step_i == 0,
            ),
            0,
            1,
        )
    )

    output_tokens = jnp.where(
        attempt_step_i == 0,
        step_state.solution[:, 0],
        action,
    )

    # Continue trajectory
    reset_pointer = jnp.where(
        curr_soln_token == output_tokens,
        pointer_correct + 1,
        jnp.ones_like(pointer_correct, dtype=int), # Assume first token of solution is <RESET>
    )
    pointer_correct = jnp.where(
        is_prompt,
        pointer_correct,
        reset_pointer,
    )

    solution_found = jnp.logical_or(
        solution_found,
        pointer_correct == step_state.solution_len
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
    solution_found = step_state.solution_found.at[:, step_i + 1].set(solution_found)
    pointer_correct = step_state.pointer_correct.at[:, step_i + 1].set(pointer_correct)
    eos = step_state.eos.at[:, step_i + 1].set(eos)

    step_state = StepState(
        graphdef=step_state.graphdef,
        rest=step_state.rest,
        cache=cache,
        eos_token=step_state.eos_token,
        rng=rng,
        observations=observations,
        actions=actions,
        solution=step_state.solution,
        solution_len=step_state.solution_len,
        solution_found=solution_found,
        pointer_correct=pointer_correct,
        last_prompt_idx=step_state.last_prompt_idx,
        eos=eos,
        attempt_step_i=attempt_step_i,
        attempt_length=step_state.attempt_length,
        pred_mask=pred_mask,
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
    attempt_length: int,
    deterministic: int = 0,
    correct_aware_shift: int = 0,
    max_token_id_to_shift: int = 0,
):
    question = batch["sequence"]
    solution = batch["target"]
    pointer_correct = batch["pointer_correct"]
    question_len = batch["question_len"]
    solution_len = batch["solution_len"]
    last_prompt_idx = jnp.argmax(batch["mask"], axis=-1)
    num_questions, max_step = question.shape

    step_state = StepState(
        graphdef=graphdef,
        rest=rest,
        cache=cache,
        eos_token=eos_token,
        rng=rng,
        observations=question,
        actions=jnp.zeros_like(question, dtype=int),
        solution=solution,
        solution_len=solution_len,
        solution_found=jnp.zeros_like(question, dtype=bool),
        pointer_correct=jnp.full(
            (num_questions, max_step),
            fill_value=-1,
            dtype=int,
        ).at[:, 0].set(pointer_correct),
        last_prompt_idx=last_prompt_idx,
        eos=jnp.zeros((num_questions, max_step), dtype=bool),
        attempt_step_i=jnp.zeros((num_questions), dtype=int),
        attempt_length=attempt_length,
        pred_mask=jnp.zeros((num_questions, max_step), dtype=bool),
        correct_aware_shift=correct_aware_shift,
        max_token_id_to_shift=max_token_id_to_shift,
        deterministic=deterministic,
    )

    step_state = jax.lax.while_loop(
        lambda state: jnp.logical_and(
            jnp.logical_not(jnp.all(state.solution_found[:, state.step_i])),
            state.step_i < max_step - 1,
        ),
        predict_step,
        step_state,
    )

    success = jnp.max(step_state.solution_found, axis=-1)
    solution_found = jnp.cumsum(step_state.solution_found, axis=-1) > 0
    solution_found_mask = success == 1
    response_length = (
        solution_found_mask * (jnp.argmax(solution_found, axis=-1) - question_len)
        + (1 - solution_found_mask) * (max_step - question_len - 1)
    )

    return RolloutResult(
        observations=step_state.observations[:, :-1],
        actions=step_state.actions[:, 1:],
        solution_found=solution_found[:, 1:],
        success=success,
        response_length=response_length,
        pred_mask=step_state.pred_mask[:, :-1],
        pointer_correct=step_state.pointer_correct[:, 1:],
        last_prompt_idx=step_state.last_prompt_idx,
        eos=step_state.eos[:, 1:],
    )
