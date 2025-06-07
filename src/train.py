import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dill
import math
import numpy as np
import tqdm

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

import src.learner as learners

from src.constants import *
from src.utils import DummySummaryWriter


def train(
    config: SimpleNamespace,
    hyperparameter_str: str,
    save_path: str = None,
):
    """
    Executes the training loop.

    :param config: the experiment configuration
    :param hyperparameter_str: the hyperparameter setting in string format
    :param save_path: the directory to save the experiment progress
    :type learner: Learner
    :type config: SimpleNamespace
    :type hyperparameter_str: str
    :type save_path: str: (Default value = None)

    """
    logging_config = config.logging_config

    num_digits = int(math.log10(config.num_epochs)) + 1

    def pad_string(s):
        s = str(s)
        return "0" * (num_digits - len(s)) + s

    true_epoch = 0
    summary_writer = DummySummaryWriter()
    try:
        learner = getattr(learners, config.learner)(config)

        if save_path:
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
            summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
            summary_writer.add_text(
                CONST_HYPERPARAMETERS,
                hyperparameter_str,
            )
            dill.dump(
                learner.state,
                open(
                    os.path.join(save_path, "models", "{}.dill".format(pad_string(0))),
                    "wb",
                ),
            )

        for epoch in tqdm.tqdm(range(config.num_epochs)):
            train_aux = learner.update(epoch)
            true_epoch = epoch + 1

            if (
                save_path
                and logging_config.log_interval
                and true_epoch % logging_config.log_interval == 0
            ):
                # NOTE: we expect the user to properly define the logging scalars in the learner
                for key, val in train_aux.items():
                    if key.startswith("hist/"):
                        summary_writer.add_text(
                            key,
                            ",".join([f"{k}: {v}" for k, v in zip(*np.unique(val, return_counts=True))]),
                            true_epoch,
                        )
                    else:
                        summary_writer.add_scalar(key, val, true_epoch)

            if (
                save_path
                and logging_config.checkpoint_interval
                and (true_epoch % logging_config.checkpoint_interval == 0)
            ):

                dill.dump(
                    learner.state,
                    open(
                        os.path.join(
                            save_path,
                            "models",
                            "{}.dill".format(pad_string(true_epoch)),
                        ),
                        "wb",
                    ),
                )
    except KeyboardInterrupt:
        pass

    if save_path:
        dill.dump(
            learner.state,
            open(
                os.path.join(
                    save_path, "models", "{}.dill".format(pad_string(true_epoch))
                ),
                "wb",
            ),
        )

    learner.close()
