import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np
import timeit


class ICLinearRegression(IterableDataset):
    def __init__(
        self,
        num_tasks: int,
        num_dims: int,
        context_len: int,
        seed: int,
        train: bool,
        input_noise_std: float = 0.0,
        label_noise_std: float = 0.0,
    ):
        self.num_tasks = num_tasks
        self.num_dims = num_dims
        self.context_len = context_len
        self.train = train
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.label_noise_std = label_noise_std
        self.rng = np.random.RandomState(seed)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(1,))

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**16) + int(self.train))

        if self.num_tasks is not None:
            def sample_weights(rng):
                task_i = rng.choice(self.num_tasks)

                task_rng = np.random.RandomState(task_i)
                weights = task_rng.standard_normal((self.num_dims, 1))
                return weights
        else:
            def sample_weights(rng):
                return rng.standard_normal((self.num_dims, 1))

        while True:
            # Sample task
            weights = sample_weights(sample_rng)

            # Inputs
            inputs = self.rng.standard_normal(size=(self.context_len, self.num_dims))
            inputs += self.rng.standard_normal(inputs.shape) * self.input_noise_std
            inputs /= np.linalg.norm(inputs, axis=-1, keepdims=True)

            # Targets
            targets = inputs @ weights
            targets += self.rng.standard_normal(targets.shape) * self.label_noise_std

            yield {
                "example": inputs,
                "target": targets,
            }
