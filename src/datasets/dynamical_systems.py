import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functools import reduce
from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np

class LinearSystem(IterableDataset):
    def __init__(
        self,
        context_len: int,
        num_dims: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
        show_A: bool=False,
    ):
        assert context_len >= 0
        self.context_len = context_len
        self.num_dims = num_dims
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
        self.show_A = show_A

        self._rng = np.random.RandomState(seed)
        self.sample_dynamics()

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, (self.num_dims,))

    @property
    def output_space(self):
        return spaces.Box(-np.inf, np.inf, (self.num_dims,))

    def __iter__(self):
        return iter(self.get_sequences())

    def sample_dynamics(self):
        self.A = np.diag(self._rng.randn(self.num_dims,))

        if self.sequence_type.startswith("heldout_A"):
            std = float(self.sequence_type.split(":")[-1])
            self.A = np.diag(self._rng.randn(self.num_dims,)) * std

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )
        while True:
            x_0 = sample_rng.randn(self.num_dims)

            if self.sequence_type.startswith("shifted_x"):
                shift = float(self.sequence_type.split(":")[1])
                x_0 += shift

            if self.sequence_type == "ic_only":
                A = np.diag(sample_rng.randn(self.num_dims,))
            else:
                A = self.A

            sequence = [x_0]
            for _ in range(self.context_len + 1):
                sequence.append(
                    A @ sequence[-1]
                )

            if self.show_A:
                sequence = list(A) + sequence

            yield {
                "sequence": np.array(sequence[:-1]),
                "target": np.array(sequence[1:]),
                "mask": np.eye(len(sequence) - 1)[-1][:, None],
            }
