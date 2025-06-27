import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np


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
        sparsity: int = -1,
        target_generator: str = "ground_truth",
    ):
        assert sparsity <= num_dims

        self.num_tasks = num_tasks
        self.num_dims = num_dims
        self.context_len = context_len
        self.train = train
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.label_noise_std = label_noise_std
        self.sparsity = sparsity
        self.target_generator = target_generator
        self.rng = np.random.RandomState(seed)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(1,))

    def __iter__(self):
        return iter(self.get_sequences())

    def get_weight_sampler(self):
        if self.num_tasks is not None:
            def sample_weights(rng):
                task_i = rng.choice(self.num_tasks)

                task_rng = np.random.RandomState(task_i)
                weights = task_rng.standard_normal((self.num_dims, 1))
                return weights
        else:
            def sample_weights(rng):
                return rng.standard_normal((self.num_dims, 1))
        return sample_weights

    def sparsify_weights(self, weights):
        if self.sparsity > 0:
            weights[self.sparsity:] = 0.0
        return weights
    
    def get_target_generator(self):
        if self.target_generator == "ground_truth":
            return lambda inputs, weights: inputs @ weights
        elif self.target_generator == "closed_form":
            def get_target(inputs, weights):
                outputs = inputs @ weights
                return np.linalg.lstsq(
                    inputs,
                    outputs,
                    rcond=None,
                )[0]
            return get_target
        elif self.target_generator == "weight_retrieval":
            pretrained_weights = np.concatenate([
                np.random.RandomState(task_i).standard_normal(
                    (self.num_dims, 1)
                )
                for task_i in range(self.num_tasks)
            ], axis=-1)

            def get_target(inputs, weights):
                closest_task = np.argmin(
                    np.sum((pretrained_weights - weights) ** 2, axis=0)
                )

                return inputs @ pretrained_weights[:, closest_task]
            return get_target
        elif self.target_generator == "weighted_weights":
            pretrained_weights = np.concatenate([
                np.random.RandomState(task_i).standard_normal(
                    (self.num_dims, 1)
                )
                for task_i in range(self.num_tasks)
            ], axis=-1)

            def get_target(inputs, weights):
                task_dists = np.exp(
                    - np.sum(
                        (pretrained_weights - weights) ** 2,
                        axis=0,
                    )
                )
                task_dists /= np.sum(task_dists)
                weighted_weights = pretrained_weights * task_dists[None]
                return np.sum(inputs @ weighted_weights, axis=1, keepdims=True)

            return get_target
        else:
            raise NotImplementedError

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**16) + int(self.train))
        sample_weights = self.get_weight_sampler()
        target_generator = self.get_target_generator()

        while True:
            # Sample task
            weights = sample_weights(sample_rng)
            weights = self.sparsify_weights(weights)

            # Inputs
            inputs = self.rng.standard_normal(size=(self.context_len, self.num_dims))
            inputs += self.rng.standard_normal(inputs.shape) * self.input_noise_std
            inputs /= np.linalg.norm(inputs, axis=-1, keepdims=True)

            # Targets
            targets = target_generator(inputs, weights)
            targets += self.rng.standard_normal(targets.shape) * self.label_noise_std

            yield {
                "weights": weights,
                "target": targets,
                "example": inputs,
            }
