import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from types import SimpleNamespace
from typing import Any, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from src.constants import *


def linear_warmup_sqrt_decay(
    max_lr: chex.Scalar,
    warmup_steps: int,
) -> optax.Schedule:
    """
    Returns a scheduler that performs linear warmup followed by an inverse square root decay of learning rate.
    Reference: https://arxiv.org/pdf/2205.05055.pdf
    """
    assert max_lr > 0, "maximum learning rate {} must be positive".format(max_lr)
    assert warmup_steps > 0, "warm up steps {} must be positive".format(warmup_steps)

    def schedule(count):
        """Linear warmup and then an inverse square root decay of learning rate."""
        linear_ratio = max_lr / warmup_steps
        decay_ratio = jnp.power(warmup_steps * 1.0, 0.5) * max_lr
        return jnp.min(
            jnp.array(
                [linear_ratio * (count + 1), decay_ratio * jnp.power((count + 1), -0.5)]
            )
        )

    return schedule


def get_scheduler(
    scheduler_config: SimpleNamespace,
) -> optax.Schedule:
    """
    Gets a scheduler.

    :param scheduler_config: the scheduler configuration
    :type scheduler_config: SimpleNamespace
    :return: the scheduler
    :rtype: optax.Schedule
    """
    kwargs = scheduler_config.scheduler_kwargs
    if scheduler_config.scheduler == CONST_LINEAR_WARMUP_SQRT_DECAY:
        return linear_warmup_sqrt_decay(
            kwargs.max_lr,
            kwargs.warmup_steps,
        )
    else:
        return getattr(optax, scheduler_config.scheduler)(**vars(kwargs))


def get_optimizer(
    opt_config: SimpleNamespace,
) -> Union[
    Tuple[Dict[str, Any], Dict[str, Any]],
    Tuple[optax.GradientTransformation, optax.OptState],
]:
    """
    Gets an optimizer and its optimizer state.

    :param opt_config: the optimizer configuration
    :type opt_config: SimpleNamespace
    :return: an optimizer and its optimizer state
    :rtype: Union[
        Tuple[Dict[str, Any], Dict[str, Any]],
        Tuple[optax.GradientTransformation, optax.OptState]
    ]

    """

    opt_transforms = []
    if opt_config.optimizer == CONST_FROZEN:
        opt_transforms.append(optax.set_to_zero())
    else:
        if opt_config.max_grad_norm:
            opt_transforms.append(optax.clip_by_global_norm(opt_config.max_grad_norm))

        opt_transforms.append(
            optax.inject_hyperparams(getattr(optax, opt_config.optimizer))(
                get_scheduler(opt_config.lr), **vars(opt_config.opt_kwargs)
            )
        )

    opt = optax.chain(*opt_transforms)

    return opt
