import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import math
import numpy as np


class KParity(IterableDataset):
    def __init__(
        self,
        sequence_length: int,
        k: int,
        train: bool,
        seed: int,
        sequence_type: str="no_cot",
    ):
        assert sequence_length >= k > 0
        self._rng = np.random.RandomState(seed)
        self.seed = seed
        self.train = train
        self.sequence_length = sequence_length
        self.k = k
        self.sequence_type = sequence_type

        self.get_train_sequences()

    @property
    def input_space(self):
        # 0, 1, <EOS>
        return spaces.Discrete(3)

    @property
    def output_space(self):
        return spaces.Discrete(3)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_k_indices(self):
        return np.arange(self.sequence_length)[:self.k]

    def get_train_sequences(self):
        # ~80% of training sequences
        self.is_train_sequence = self._rng.binomial(
            1,
            0.8,
            size=(2 ** self.sequence_length,)
        )
        if self.train:
            self.sequence_indices = np.where(self.is_train_sequence == 1)[0]
        else:
            self.sequence_indices = np.where(self.is_train_sequence == 0)[0]

    def generate_cot(self, sequence_int, k_indices):
        if self.sequence_type == "no_cot":
            return sequence_int
        elif self.sequence_type == "full_cot":
            curr_sum = 0
            for idx in k_indices:
                curr_sum = (curr_sum + sequence_int[idx]) % 2
                sequence_int.append(curr_sum)
            return sequence_int
        elif self.sequence_type == "partial_cot":
            curr_sum = 0
            for idx in k_indices[:len(k_indices) // 2]:
                curr_sum = (curr_sum + sequence_int[idx]) % 2
                sequence_int.append(curr_sum)
            return sequence_int
        else:
            raise NotImplementedError

    def get_sequences(self):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )
        while True:
            k_indices = self.get_k_indices()
            sequence_id = sample_rng.choice(self.sequence_indices)
            sequence_str = "{0:b}".format(sequence_id)
            sequence_str = sequence_str.rjust(self.sequence_length, "0")

            sequence_int = [int(token_id) for token_id in sequence_str] + [2]
            target = sum(sequence_int[idx] for idx in k_indices) % 2

            sequence_int = self.generate_cot(
                sequence_int,
                k_indices,
            )

            sequence_int.append(target)
            sequence_int = np.array(sequence_int)

            yield {
                "task": np.array(sequence_id),
                "target": sequence_int[1:],
                "sequence": sequence_int[:-1],
            }
