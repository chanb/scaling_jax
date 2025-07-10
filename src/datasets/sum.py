import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np

"""
TODO:
Context sequence:
t_1, t_2, ..., t_L, t,
where
- t_l in {0, 1}
- t in {0, 2^L - 1}
- t_1 + ... + t_L = t

Goal:
- Predict t >= C, for some C in {0, 2^L - 1}

TODO:
- For now, assume C is half of the set
    - Model can exploit this easily for in-context since it can just look at half the bits
"""
class ThresholdSum(IterableDataset):
    def __init__(
        self,
        context_len: int,
        train: bool,
        seed: int,
        train_val_ratio: float=0.8,
        include_boundary: bool=True,
    ):
        assert context_len > 0
        assert 0 < train_val_ratio < 1
        self.context_len = context_len
        self.num_elements = int(2 ** context_len)
        self.train = train
        self.seed = seed
        self.train_val_ratio = train_val_ratio
        self.include_boundary = include_boundary
        self.threshold = self.num_elements // 2

        self._rng = np.random.RandomState(seed)
        self.get_train_sequences()

    @property
    def input_space(self):
        return spaces.Discrete(self.num_elements + 2)

    @property
    def output_space(self):
        return spaces.Discrete(self.num_elements + 2)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_train_sequences(self):
        # Permute IDs, first partition is for train and second partition is for test
        self.sequence_indices = self._rng.permutation(self.num_elements)
        num_train = int(np.floor(self.num_elements * self.train_val_ratio))

        if self.include_boundary:

            # Make sure C and C - 1 are in training sequence
            midpoint_idx = np.where(self.sequence_indices == self.threshold)[0][0]
            (self.sequence_indices[0], self.sequence_indices[midpoint_idx]) = (
                self.sequence_indices[midpoint_idx], self.sequence_indices[0]
            )

            midpoint_idx = np.where(self.sequence_indices == self.threshold - 1)[0][0]
            (self.sequence_indices[1], self.sequence_indices[midpoint_idx]) = (
                self.sequence_indices[midpoint_idx], self.sequence_indices[1]
            )

        if self.train:
            self.sequence_indices = self.sequence_indices[:num_train]
        else:
            self.sequence_indices = self.sequence_indices[num_train:]

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )
        while True:
            t = sample_rng.choice(self.sequence_indices)
            bin_repr = "{0:b}".format(t)
            bin_repr = bin_repr.rjust(self.context_len, "0")

            # Last two token IDs are for binary representation
            sequence = [
                int(token_id) + self.num_elements
                for token_id in bin_repr
            ] + [t]

            target = sequence[:-1] + [int(t >= self.threshold) + self.num_elements]

            yield {
                "sequence": np.array(sequence),
                "target": np.array(target),
                "mask": np.eye(len(sequence))[-1],
            }
