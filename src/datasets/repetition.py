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


class Repetition(IterableDataset):
    def __init__(
        self,
        context_len: int,
        vocab_size: int,
        k: int,
        train: bool,
        seed: int,
        sequence_type: str="default",
        train_val_ratio: float=0.8,
        num_repeats: int=None,
        shuffle: bool=True,
        exact: bool=False,
        num_cot_tokens: int=0,
    ):
        assert context_len > 0
        assert vocab_size > 1
        assert 0 < train_val_ratio <= 1
        assert k >= 0
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.k = k
        self.train = train
        self.seed = seed
        self.sequence_type = sequence_type
        self.train_val_ratio = train_val_ratio
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.exact = exact
        self.num_cot_tokens = num_cot_tokens
        self._eos_token_id = (
            1 # RESET
            + self.vocab_size
            + self.num_cot_tokens
        )

        self._rng = np.random.RandomState(seed)
        self.get_train_sequences()

        print("EOS TOKEN: {}".format(self.eos_token_id))
        print("TOKEN MAP: {}".format(self.token_map))

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def max_token_id_to_shift(self):
        return -1

    @property
    def correctness_aware_tokens_offset(self):
        return -1

    @property
    def reset_token_id(self):
        return self.vocab_size + self.num_cot_tokens

    @property
    def token_map(self):
        base_token_map = {
            **{
                vocab_i: vocab_i
                for vocab_i in range(self.vocab_size)
            },
            self.reset_token_id: self.reset_token_id,
            **{
                cot_token_id + self.vocab_size:
                cot_token_id + self.vocab_size
                for cot_token_id in range(self.num_cot_tokens)
            },
            self.eos_token_id: self.eos_token_id,
        }

        return base_token_map

    @property
    def input_space(self):
        # 0, 1, 2, ..., K - 1, <RESET>, [<REG_1>, ..., <REG_K>], <EOS>
        return spaces.Discrete(self.vocab_size + 1 + self.num_cot_tokens + 1)

    @property
    def output_space(self):
        # 0, 1, 2, ..., K - 1, [<REG_1>, ..., <REG_K>]
        return spaces.Discrete(self.vocab_size + self.num_cot_tokens)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_train_sequences(self):
        # Permute IDs, first partition is for train and second partition is for test
        self.vocabs = self._rng.permutation(self.vocab_size)
        num_train = int(np.floor(self.vocab_size * self.train_val_ratio))

        if self.train:
            self.vocabs = self.vocabs[:num_train]
        else:
            self.vocabs = self.vocabs[num_train:]

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(
            self._rng.randint(0, 2**16) + int(self.train)
        )
        curr_idx = 0
        repeated = 0
        original_vocabs = self.vocabs[:]

        if self.sequence_type == "ntp":
            total_attempts = math.ceil(self.context_len / self.k)
            pre_gen_tokens = []
            for attempt_i in range(total_attempts):
                curr = attempt_i
                str_repr = ""
                for bit_i in range(self.k):
                    curr_bit = curr % self.vocab_size
                    curr = math.floor(curr / self.vocab_size)
                    str_repr = str_repr + "{},".format(curr_bit)
                str_repr = str_repr[:-1]
                pre_gen_tokens.append(
                    list(map(int, str_repr.split(",")))
                )
            print(pre_gen_tokens)
        while True:
            if (
                self.num_repeats is not None
                and len(self.vocabs) == 0
            ):
                if repeated == self.num_repeats:
                    break
                repeated += 1
                self.vocabs = original_vocabs[:]

            if self.shuffle:
                t = sample_rng.choice(self.vocabs)
                if self.num_repeats is not None:
                    self.vocabs = self.vocabs[
                        self.vocabs != t
                    ]
            else:
                t = self.vocabs[curr_idx]

                if self.num_repeats is None:
                    curr_idx = (curr_idx + 1) % len(self.vocabs)

            sequence = [t] + [self.reset_token_id] + [self.eos_token_id] * (self.context_len - 2)
            question_len = 1
            solution_len = 1 + self.k
            
            if self.sequence_type == "question_only":
                soln_list_repr = [self.reset_token_id] + [t] * self.k + [self.eos_token_id] * (self.context_len - self.k - 1)

                mask = np.zeros(len(sequence), dtype=bool)
                mask[question_len:] = True
                
                yield {
                    "sequence": np.array(sequence),
                    "target": np.array(soln_list_repr),
                    "mask": mask,
                    "pointer_correct": 1,
                    "question_len": question_len,
                    "solution_len": solution_len,
                }
            elif self.sequence_type == "ntp":
                soln_list_repr = [self.reset_token_id] + [t] * self.k

                attempt_i = 0
                for step_i in range(self.context_len - self.k):
                    attempt_step = step_i % (self.k + 1)
                    if attempt_step == 0:
                        attempt_i += 1
                        next_k_tokens = [self.reset_token_id, *pre_gen_tokens[attempt_i]]
                    soln_list_repr = soln_list_repr + [next_k_tokens[attempt_step]]

                mask = np.ones(len(soln_list_repr), dtype=bool)
                mask[:question_len] = False
                mask[::(self.k + 1)] = False
                
                yield {
                    "sequence": np.array(soln_list_repr)[:-1],
                    "target": np.array(soln_list_repr)[1:],
                    "mask": mask[1:],
                    "pointer_correct": 1,
                    "question_len": question_len,
                    "solution_len": solution_len,
                }

