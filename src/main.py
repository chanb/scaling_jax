"""
This script is the entrypoint for any experiment.
XXX: Try not to modify this.
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datetime import datetime
from pprint import pprint

import json
import os
import timeit
import uuid

from src.constants import *
from src.train import train
from src.utils import flatten_dict, parse_dict, set_seed


"""
This function constructs the model, optimizer, and learner and executes training.
"""


def main(config_path: str):
    """
    Orchestrates the experiment.

    :param config_path: the experiment configuration file path
    :type config_path: str

    """
    tic = timeit.default_timer()

    assert os.path.isfile(config_path), f"{config_path} is not a file"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        hyperparameter_str = "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in dict(flatten_dict(config_dict)).items()
                ]
            )
        )
        config = parse_dict(config_dict)

    pprint(config)

    set_seed(config.seeds.seed)
    save_path = None
    if config.logging_config.save_path:
        optional_prefix = ""
        if config.logging_config.experiment_name:
            optional_prefix += f"{config.logging_config.experiment_name}-"
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        run_id = str(uuid.uuid4())
        save_path = os.path.join(
            config.logging_config.save_path, f"{optional_prefix}{time_tag}-{run_id}"
        )
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w+") as f:
            json.dump(config_dict, f)

    train(config, hyperparameter_str, save_path)
    toc = timeit.default_timer()
    print(f"Experiment Time: {toc - tic}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="Training configuration", required=True
    )
    args = parser.parse_args()

    config_path = args.config_path
    main(config_path)
