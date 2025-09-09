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
        shuffle: bool=True,
        exact: bool=False,
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
        self.shuffle = shuffle
        self.exact = exact
        self.max_bit_len = math.ceil(np.log2(max_int))

        self._rng = np.random.RandomState(seed)
        self.get_train_sequences()

    @property
    def input_space(self):
        # 0, 1, <PLUS>, <EQUAL>, <EOS>
        return spaces.Discrete(5)

    @property
    def output_space(self):
        return spaces.Discrete(5)

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
        while True:
            if self.shuffle:
                t = sample_rng.choice(self.sequence_indices)
            else:
                t = self.sequence_indices[curr_idx]
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
        sequence_type: str="default",
        train_val_ratio: float=0.8,
        include_boundary: bool=True,
    ):
        assert context_len > 0
        assert 0 < train_val_ratio < 1
        self.context_len = context_len
        self.num_elements = int(2 ** context_len)
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
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

            exceeds_threshold = t >= self.threshold

            if self.sequence_type == "iw":
                new_t = (
                    sample_rng.randint(0, self.threshold)
                    if exceeds_threshold else
                    sample_rng.randint(self.threshold, self.num_elements)
                )
                bin_repr = "{0:b}".format(new_t)
                bin_repr = bin_repr.rjust(self.context_len, "0")
                sequence[:-1] = [
                    int(token_id) + self.num_elements
                    for token_id in bin_repr
                ]
            elif self.sequence_type == "ic":
                new_t = (
                    sample_rng.randint(0, self.threshold)
                    if exceeds_threshold else
                    sample_rng.randint(self.threshold, self.num_elements)
                )
                sequence[-1] = new_t                

            target = sequence[:-1] + [int(exceeds_threshold) + self.num_elements]
            yield {
                "sequence": np.array(sequence),
                "target": np.array(target),
                "mask": np.eye(len(sequence))[-1],
            }


class XOR(IterableDataset):
    def __init__(
        self,
        context_len: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
        train_val_ratio: float=0.8,
        include_boundary: bool=True,
    ):
        assert context_len > 0
        assert 0 < train_val_ratio < 1
        self.context_len = context_len
        self.num_elements = int(2 ** context_len)
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
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

            xor_res = reduce(
                lambda carry, curr_el: carry ^ int(curr_el),
                bin_repr,
                0
            )

            if self.sequence_type == "iw":
                while True:
                    new_t = sample_rng.randint(0, self.num_elements)
                    bin_repr = "{0:b}".format(new_t)
                    bin_repr = bin_repr.rjust(self.context_len, "0")

                    if xor_res != reduce(
                        lambda carry, curr_el: carry ^ int(curr_el),
                        bin_repr,
                        0
                    ):
                        break
                sequence[:-1] = [
                    int(token_id) + self.num_elements
                    for token_id in bin_repr
                ]
            elif self.sequence_type == "ic":
                # XXX: The heldout here is bad mostly because it learned to flip the label of the query
                while True:
                    new_t = sample_rng.randint(0, self.num_elements)
                    bin_repr = "{0:b}".format(new_t)
                    bin_repr = bin_repr.rjust(self.context_len, "0")

                    if xor_res != reduce(
                        lambda carry, curr_el: carry ^ int(curr_el),
                        bin_repr,
                        0
                    ):
                        break

                sequence[-1] = new_t

            target = [xor_res + self.num_elements] * len(sequence)
            yield {
                "sequence": np.array(sequence),
                "target": np.array(target),
                "mask": np.eye(len(sequence))[-1],
            }
