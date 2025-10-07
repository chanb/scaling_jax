import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functools import reduce
from gymnasium import spaces
from torch.utils.data import IterableDataset

import math
import numpy as np


class Addition(IterableDataset):
    def __init__(
        self,
        context_len: int,
        max_int: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
        train_val_ratio: float=0.8,
        right_to_left: bool=False,
        num_repeats: int=None,
        shuffle: bool=True,
        exact: bool=False,
        predict_eos: bool=True,
    ):
        assert context_len > 0
        assert max_int > 0
        assert 0 < train_val_ratio <= 1
        self.context_len = context_len
        self.max_int = max_int
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
        self.train_val_ratio = train_val_ratio
        self.right_to_left = right_to_left
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.exact = exact
        self.max_bit_len = math.ceil(np.log2(max_int))
        self.predict_eos = predict_eos

        self._rng = np.random.RandomState(seed)
        self.get_train_sequences()

    @property
    def input_space(self):
        # 0, 1, <PLUS>, <EQUAL>, <EOS>
        return spaces.Discrete(5)

    @property
    def output_space(self):
        return spaces.Discrete(4 + int(self.predict_eos))

    def __iter__(self):
        return iter(self.get_sequences())

    def get_train_sequences(self):
        self.num_pairs = self.max_int ** 2

        # Permute IDs, first partition is for train and second partition is for test
        self.sequence_indices = self._rng.permutation(self.num_pairs)
        num_train = int(np.floor(self.num_pairs * self.train_val_ratio))

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
        curr_idx = 0
        repeated = 0
        while True:
            if (
                self.num_repeats is not None
                and len(self.sequence_indices) == 0
            ):
                if repeated == self.num_repeats:
                    break
                repeated += 1

            if self.shuffle:
                t = sample_rng.choice(self.sequence_indices)
                if self.num_repeats is not None:
                    self.sequence_indices = self.sequence_indices[
                        self.sequence_indices != t
                    ]
            else:
                t = self.sequence_indices[curr_idx]

                if self.num_repeats is None:
                    curr_idx = (curr_idx + 1) % len(self.sequence_indices)

            # Assume equal length for both integers for now
            first_int = t // self.max_int
            second_int = t % self.max_int
            soln = first_int + second_int

            first_bin_repr = "{0:b}".format(first_int)
            second_bin_repr = "{0:b}".format(second_int)
            soln_bin_repr = "{0:b}".format(soln)


            max_len = max(len(first_bin_repr), len(second_bin_repr))
            if self.exact and max_len != self.max_bit_len:
                continue

            first_bin_repr = first_bin_repr.rjust(max_len, "0")
            second_bin_repr = second_bin_repr.rjust(max_len, "0")

            first_list_repr = [
                int(token_id)
                for token_id in first_bin_repr
            ]
            second_list_repr = [
                int(token_id)
                for token_id in second_bin_repr
            ]
            soln_list_repr = [
                int(token_id)
                for token_id in soln_bin_repr
            ]

            if self.right_to_left:
                first_list_repr = first_list_repr[::-1]
                second_list_repr = second_list_repr[::-1]
                soln_list_repr = soln_list_repr[::-1]

            sequence = first_list_repr + [2] + second_list_repr

            question_len = len(sequence)
            
            if self.sequence_type == "default":
                sequence = sequence + [3] + soln_list_repr
                sequence = sequence + [4] * (self.context_len - len(sequence) + 1)
                mask = np.zeros(len(sequence) - 1)
                mask[question_len:] = 1
        
                yield {
                    "sequence": np.array(sequence)[:-1],
                    "target": np.array(sequence)[1:],
                    "mask": mask,
                }
            elif self.sequence_type == "cot":
                sequence = sequence + [3] + soln_list_repr[::-1] + [3] + soln_list_repr
                sequence = sequence + [4] * (self.context_len - len(sequence) + 1)
                mask = np.zeros(len(sequence) - 1)
                mask[question_len:] = 1
        
                yield {
                    "sequence": np.array(sequence)[:-1],
                    "target": np.array(sequence)[1:],
                    "mask": mask,
                }
            elif self.sequence_type == "question_only_decode":
                sequence = sequence + [3]
                mask = np.ones(len(soln_list_repr))
                yield {
                    "sequence": np.array(sequence),
                    "target": np.array(soln_list_repr),
                    "mask": mask,
                }
            elif self.sequence_type == "question_only":
                sequence = sequence + [3]
                sequence = sequence + [4] * (self.context_len - len(sequence) + 1)
                soln_list_repr = [3] + soln_list_repr
                soln_list_repr = soln_list_repr + [4] * (self.context_len - len(soln_list_repr) + 1)

                mask = np.zeros(len(sequence))
                mask[question_len:] = 1
                
                yield {
                    "sequence": np.array(sequence),
                    "target": np.array(soln_list_repr),
                    "mask": mask,
                }
