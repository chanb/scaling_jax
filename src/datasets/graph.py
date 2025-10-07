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


class StarPathFinding(IterableDataset):
    def __init__(
        self,
        context_len: int,
        max_star_length: int,
        max_branches: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
        train_val_ratio: float=0.8,
        num_repeats: int=None,
        exact: bool=False,
        predict_eos: bool=True,
        num_cot_tokens: int=0,
        vocab_size: int=None,
    ):
        assert context_len > 0
        assert max_star_length > 0
        assert max_branches > 0
        assert 0 < train_val_ratio <= 1
        self.context_len = context_len
        self.max_star_length = max_star_length
        self.max_branches = max_branches
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
        self.train_val_ratio = train_val_ratio
        self.num_repeats = num_repeats
        self.exact = exact
        self.predict_eos = predict_eos
        self.num_cot_tokens = num_cot_tokens

        self.num_nodes = max_star_length * max_branches + 1
        self.vocab_size = 6 + self.num_nodes + self.num_cot_tokens
        self.eos_token_id = self.vocab_size - 1

        if vocab_size is not None and vocab_size >= self.vocab_size:
            self.vocab_size = vocab_size

        self._rng = np.random.RandomState(seed)
        self.get_train_sequences()

    @property
    def input_space(self):
        # <SOURCE>, <TARGET>, <TO>, <SEP>, <EQUAL>, <N_0>, ..., <N_K>, <REG_1>, ..., <REG_L>, <EOS>
        return spaces.Discrete(self.vocab_size)

    @property
    def output_space(self):
        return spaces.Discrete(
            self.vocab_size - int(self.predict_eos)
        )

    def __iter__(self):
        return iter(self.get_sequences())

    def get_train_sequences(self):
        pass

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )
        curr_idx = 0
        repeated = 0
        original_sequence_indices = self.sequence_indices[:]
        while True:
            pass