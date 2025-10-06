import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np


class Parity(IterableDataset):
    def __init__(
        self,
        context_len: int,
        max_level: int,
        train: bool,
        seed: int,
        train_val_ratio: float=0.8,
        exact: bool=False,
        num_cot_tokens: int=0,
        repeat: bool=True,
    ):
        assert context_len > max_level > 0
        assert 0 < train_val_ratio <= 1.0
        assert num_cot_tokens >= 0

        self._rng = np.random.RandomState(seed)
        self.seed = seed
        self.train = train
        self.context_len = context_len
        self.num_cot_tokens = num_cot_tokens
        self.train_val_ratio = train_val_ratio
        self.exact = exact
        self.max_level = max_level
        self.repeat = repeat

        self.get_train_sequences()

    @property
    def input_space(self):
        # 0, 1, <?>, <EOS>, <REG_1>, ..., <REG_K>
        return spaces.Discrete(4 + self.num_cot_tokens)

    @property
    def output_space(self):
        return spaces.Discrete(4 + self.num_cot_tokens)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_train_sequences(self):
        # train_val_ratio of training sequences
        if self.exact:
            self.num_elements = 2 ** self.max_level

            def id_to_bin_str(sequence_id):
                bin_str = np.base_repr(
                    sequence_id,
                    base=2,
                ).zfill(self.max_level - 1)

                return bin_str

        else:
            n_ary = 2
            num_levels = self.max_level - 1
            self.num_elements = int((n_ary ** (num_levels + 1) - 1) / (n_ary - 1))

            def id_to_bin_str(sequence_id):
                num_bits = 1
                while True:
                    num_elements_at_level = 2 ** num_bits
                    if sequence_id < num_elements_at_level:
                        break
                    sequence_id -= num_elements_at_level
                    num_bits += 1

                bin_str = np.base_repr(
                    sequence_id,
                    base=2,
                ).zfill(num_bits)

                return bin_str

        self.id_to_bin_str = id_to_bin_str

        self.is_train_sequence = self._rng.binomial(
            1,
            self.train_val_ratio,
            size=(self.num_elements,)
        )

        if self.train:
            self.sequence_indices = np.where(self.is_train_sequence == 1)[0]
        else:
            self.sequence_indices = np.where(self.is_train_sequence == 0)[0]

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )

        while True:
            if not self.repeat and len(self.sequence_indices) == 0:
                break

            sequence_id = sample_rng.choice(self.sequence_indices)

            if not self.repeat:
                self.sequence_indices = self.sequence_indices[
                    self.sequence_indices != sequence_id
                ]

            bin_str = self.id_to_bin_str(sequence_id)
            sequence = [
                int(token_id)
                for token_id in bin_str
            ]
            target = sum(sequence) % 2

            question_len = len(sequence)

            # Construct sequence
            sequence = sequence + [3]
            sequence = sequence + [4] * (self.context_len - len(sequence) + 1)

            soln_list_repr = [3, target]
            soln_list_repr = soln_list_repr + [4] * (self.context_len - len(soln_list_repr) + 1)

            mask = np.zeros(len(sequence))
            mask[question_len:] = 1
            
            yield {
                "sequence": np.array(sequence),
                "target": np.array(soln_list_repr),
                "mask": mask,
            }