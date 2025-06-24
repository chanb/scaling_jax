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


def n_str_to_id(n_ary, n_str):
    level = len(n_str)

    # (x^(n + 1) - 1) / (x - 1)
    start_ind = (n_ary ** level - 1) / (n_ary - 1)
    str_id = start_ind + int(n_str, n_ary)

    return int(str_id)


def id_to_n_str(n_ary, token_id):
    if token_id == 0:
        return "<EOS>"

    level = 0
    start_ind = n_ary ** level

    while True:
        if (token_id - start_ind) < 0:
            break
        level += 1
        start_ind += n_ary ** level

    remainder = token_id + int(n_ary ** level) - start_ind

    n_str = ""
    while remainder > 0:
        n_str = str(remainder % n_ary) + n_str
        remainder //= n_ary

    n_str = n_str.rjust(level, "0")
    return n_str


def np_to_n_str(arr):
    return "".join(map(str, arr))


class NaryStrings(IterableDataset):
    def __init__(
        self,
        n_ary: int,
        num_levels: int,
        context_len: int,
        min_sequence_len: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
    ):
        assert num_levels > 0
        assert n_ary >= 2
        self._rng = np.random.RandomState(seed + int(train))
        self.seed = seed
        self.n_ary = n_ary
        self.num_levels = num_levels
        self.context_len = context_len
        self.min_sequence_len = min_sequence_len
        self.train = train
        self.sequence_type = sequence_type

        # num elements = (x^(n + 1) - 1) / (x - 1)
        self.num_elements = int((n_ary ** (num_levels + 1) - 1) / (n_ary - 1))

    @property
    def input_space(self):
        return spaces.Discrete(self.num_elements)

    @property
    def output_space(self):
        return spaces.Discrete(self.num_elements)

    def generate_descendent_supervised_pairs(self):
        input_level = self._rng.randint(self.num_levels - 1)
        output_level = self._rng.randint(input_level + 1, self.num_levels)

        bin_strs = self._rng.randint(0, self.n_ary, size=(self.min_sequence_len, output_level + 1))

        sequence = []

        for bin_str in bin_strs:
            sequence.append(n_str_to_id(self.n_ary, np_to_n_str(bin_str[:input_level + 1])))
            sequence.append(n_str_to_id(self.n_ary, np_to_n_str(bin_str[:output_level + 1])))

        return sequence

    def generate_ascentdent_supervised_pairs(self):
        input_level = self._rng.randint(1, self.num_levels)
        output_level = self._rng.randint(input_level)

        bin_strs = self._rng.randint(0, self.n_ary, size=(self.min_sequence_len, input_level + 1))

        sequence = []

        for bin_str in bin_strs:
            sequence.append(n_str_to_id(self.n_ary, np_to_n_str(bin_str[:input_level + 1])))
            sequence.append(n_str_to_id(self.n_ary, np_to_n_str(bin_str[:output_level + 1])))

        return sequence

    def generate_descendent_chain(self):
        start_level = self._rng.randint(1, self.num_levels + 1)

        bin_str = np_to_n_str(self._rng.randint(0, self.n_ary, size=(start_level,)))

        sequence = []

        for ind_i in range(self.min_sequence_len):
            sequence.append(n_str_to_id(self.n_ary, bin_str))

            if len(bin_str) == self.num_levels:
                bin_str = ""
            bin_str += str(self._rng.randint(0, self.n_ary))

        return sequence

    def generate_ascendent_chain(self):
        start_level = self._rng.randint(1, self.num_levels + 1)

        bin_str = np_to_n_str(self._rng.randint(0, self.n_ary, size=(start_level,)))

        sequence = []

        for ind_i in range(self.min_sequence_len):
            sequence.append(n_str_to_id(self.n_ary, bin_str))

            bin_str = bin_str[:-1]
            if len(bin_str) == 0:
                bin_str = np_to_n_str(
                    self._rng.randint(0, self.n_ary, size=(self.num_levels,))
                )

        return sequence

    def generate_iid_analogy(self):
        change_idx = self._rng.randint(0, self.num_levels)

        sequence = []
        for ind_i in range(math.ceil(self.min_sequence_len / 2)):
            start_level = self._rng.randint(change_idx + 1, self.num_levels + 1)
            bin_str = np_to_n_str(self._rng.randint(0, self.n_ary, size=(start_level,)))
            sequence.append(n_str_to_id(self.n_ary, bin_str))

            bin_str = list(bin_str)
            bin_str[change_idx] = str(self._rng.randint(0, self.n_ary))
            bin_str = "".join(bin_str)
            
            sequence.append(n_str_to_id(self.n_ary, bin_str))

        return sequence

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(
        self,
    ):
        while True:
            if self.sequence_type == "default":
                task = self._rng.randint(5)
            elif self.sequence_type == "iid_asc":
                task = 0
            elif self.sequence_type == "iid_dsc":
                task = 1
            elif self.sequence_type == "chain_asc":
                task = 2
            elif self.sequence_type == "chain_dsc":
                task = 3
            elif self.sequence_type == "iid_analogy":
                task = 4
            else:
                raise NotImplementedError

            """
            TODO: For ambiguous prediction, perhaps have the previous chain be giving hints
            """
            if task == 0:
                sequence = self.generate_ascentdent_supervised_pairs()
            elif task == 1:
                sequence = self.generate_descendent_supervised_pairs()
            elif task == 2:
                sequence = self.generate_ascendent_chain()
            elif task == 3:
                sequence = self.generate_descendent_chain()
            elif task == 4:
                sequence = self.generate_iid_analogy()
            else:
                raise NotImplementedError

            sequence = sequence + [0]
            if len(sequence) > self.context_len + 1:
                start_ind = self._rng.randint(len(sequence) - self.context_len)
                sequence = sequence[start_ind:start_ind + self.context_len + 1]
            elif len(sequence) < self.context_len + 1:
                sequence.extend([0] * (self.context_len - len(sequence) + 1))

            try:
                assert np.all([seq_i <= self.num_elements for seq_i in sequence])
            except:
                import ipdb
                ipdb.set_trace()

            yield {
                "task": task,
                "target": sequence[1:],
                "sequence": sequence[:-1],
            }
