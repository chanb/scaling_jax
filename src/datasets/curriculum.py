import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.utils.data import IterableDataset

import numpy as np


class Curriculum(IterableDataset):
    def __init__(
        self,
        datasets: list,
        curriculum_schedule: list,
        curriculum_type: str,
        seed: int,
    ):
        self._rng = np.random.RandomState(seed)
        self.seed = seed
        self.datasets = datasets
        self.datasets_iters = [iter(dataset) for dataset in datasets]
        self.curriculum_schedule = curriculum_schedule
        self.curriculum_type = curriculum_type

    @property
    def input_space(self):
        return self.datasets[0].input_space

    @property
    def output_space(self):
        return self.datasets[0].output_space

    def __iter__(self):
        return iter(self.get_sequences())

    def get_dataset_to_sample(self):
        if self.curriculum_type == "hard_switch":
            def next_dataset(curriculum_i):
                return self.datasets_iters[curriculum_i]
        elif self.curriculum_type == "uniform":
            def next_dataset(curriculum_i):
                return self._rng.choice(self.datasets_iters[:curriculum_i + 1])
        else:
            raise NotImplementedError
        return next_dataset

    def get_sequences(self):
        curriculum_i = 0
        step = 0
        next_dataset_generator = self.get_dataset_to_sample()

        while True:
            step += 1
            dataset = next_dataset_generator(curriculum_i)
            yield next(dataset)

            if step >= self.curriculum_schedule[curriculum_i]:
                step = 0
                curriculum_i = max(curriculum_i + 1, len(self.datasets) - 1)
