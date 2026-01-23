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
        num_cot_tokens: int=0,
        p_curriculum: float=0.0,
        p_inject_noop: float=0.0,
        max_noops: int=0,
        noop_as_pad: bool=False,
        reverse_curriculum: bool=False,
        correctness_aware: bool=False,
        carry_registers: bool=False,
        match_carry: bool=False,
    ):
        assert context_len > 0
        assert max_int > 0
        assert 0 < train_val_ratio <= 1
        assert 0.0 <= p_inject_noop < 1.0
        assert 0.0 <= p_curriculum < 1.0
        assert max_noops >= 0
        assert not (correctness_aware and carry_registers)
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
        self.num_cot_tokens = num_cot_tokens
        self.p_curriculum = p_curriculum
        self.p_inject_noop = p_inject_noop
        self.max_noops = max_noops
        self.noop_as_pad = noop_as_pad
        self.reverse_curriculum = reverse_curriculum
        self.correctness_aware = correctness_aware
        self.carry_registers = carry_registers
        self.match_carry = match_carry
        self._eos_token_id = (
            4
            + self.num_cot_tokens
            + 2 * int(carry_registers)
            + 2 * int(correctness_aware)
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
        if self.correctness_aware:
            return 1
        return -1

    @property
    def correctness_aware_tokens_offset(self):
        return 4 + self.num_cot_tokens

    @property
    def reset_token_id(self):
        return 3

    @property
    def token_map(self):
        base_token_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            **{
                cot_token_id + 4: cot_token_id + 4 for cot_token_id in range(self.num_cot_tokens)
            },
            self._eos_token_id: self._eos_token_id,
        }

        if self.carry_registers:
            if self.match_carry:
                base_token_map[self.correctness_aware_tokens_offset] = self.correctness_aware_tokens_offset
                base_token_map[self.correctness_aware_tokens_offset + 1] = self.correctness_aware_tokens_offset + 1
            else:
                base_token_map[self.correctness_aware_tokens_offset] = 0
                base_token_map[self.correctness_aware_tokens_offset + 1] = 1
        elif self.correctness_aware:
            base_token_map[self.correctness_aware_tokens_offset] = self.correctness_aware_tokens_offset
            base_token_map[self.correctness_aware_tokens_offset + 1] = self.correctness_aware_tokens_offset + 1

        return base_token_map

    @property
    def input_space(self):
        # 0, 1, <PLUS>, <EQUAL>, [<REG_1>, ..., <REG_K>], <EOS>, [0', 1']
        return spaces.Discrete(5 + 2 * int(self.correctness_aware) + self.num_cot_tokens + 2 * int(self.carry_registers))

    @property
    def output_space(self):
        # 0, 1, <PLUS>, <EQUAL>, [<REG_1>, ..., <REG_K>], <EOS>, [0', 1']
        return spaces.Discrete(4 + int(self.predict_eos) + self.num_cot_tokens + 2 * int(self.carry_registers))

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
        original_sequence_indices = self.sequence_indices[:]
        while True:
            if (
                self.num_repeats is not None
                and len(self.sequence_indices) == 0
            ):
                if repeated == self.num_repeats:
                    break
                repeated += 1
                self.sequence_indices = original_sequence_indices[:]
                print("Repeating the dataset {}/{}".format(repeated, self.num_repeats))

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

            first_bin_repr = "{0:b}".format(first_int)
            second_bin_repr = "{0:b}".format(second_int)

            max_len = max(len(first_bin_repr), len(second_bin_repr))
            if self.exact and max_len != self.max_bit_len:
                continue

            first_bin_repr = first_bin_repr.rjust(max_len, "0")
            second_bin_repr = second_bin_repr.rjust(max_len, "0")

            if self.match_carry:
                carry = False
                soln_bin_repr = ""
                for first_bit, second_bit in zip(
                    first_bin_repr[::-1],
                    second_bin_repr[::-1],
                ):
                    curr_res = int(first_bit) + int(second_bit) + carry
                    carry = curr_res >= 2
                    soln_bin_repr = str(
                        curr_res % 2
                        + (int(carry) * self.correctness_aware_tokens_offset)
                    ) + soln_bin_repr
                if carry:
                    soln_bin_repr = "1" + soln_bin_repr
            else:
                soln = first_int + second_int
                soln_bin_repr = "{0:b}".format(soln)

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
            solution_len = len(soln_list_repr) + int(self.predict_eos)

            # TODO: Write this outside so don't need to recheck if's
            if (
                self.num_cot_tokens > 0
                and self.p_inject_noop > 0.0
                and self.max_noops > 0
            ):
                num_noops = sample_rng.binomial(self.max_noops, self.p_inject_noop)
                for _ in range(num_noops):
                    token_to_add = [4 + sample_rng.choice(self.num_cot_tokens)]

                    if self.noop_as_pad:
                        noop_position = sample_rng.randint(max_len + 1)
                        if noop_position == max_len:
                            sequence = first_list_repr + token_to_add + [2] + second_list_repr + token_to_add
                        else:
                            sequence = (
                                first_list_repr[:noop_position]
                                + token_to_add + first_list_repr[noop_position:]
                                + [2]
                                + second_list_repr[:noop_position]
                                + token_to_add + second_list_repr[noop_position:]
                            )
                        question_len += 2
                    else:
                        noop_position = sample_rng.randint(question_len + 1)
                        if noop_position == question_len:
                            sequence = sequence + token_to_add
                        else:
                            sequence = sequence[:noop_position] + token_to_add + sequence[noop_position:]
                        question_len += 1
            
            if self.sequence_type == "default":
                sequence = sequence + [3] + soln_list_repr
                sequence = sequence + [self.eos_token_id - int(not self.predict_eos)] * (self.context_len - len(sequence) + 1)
                mask = np.zeros(len(sequence) - 1)
                mask[question_len:] = 1
        
                yield {
                    "sequence": np.array(sequence)[:-1],
                    "target": np.array(sequence)[1:],
                    "mask": mask,
                    "question_len": question_len,
                    "solution_len": solution_len,
                }
            elif self.sequence_type == "cot":
                sequence = sequence + [3] + soln_list_repr[::-1] + [3] + soln_list_repr
                sequence = sequence + [self.eos_token_id - int(not self.predict_eos)] * (self.context_len - len(sequence) + 1)
                mask = np.zeros(len(sequence) - 1)
                mask[question_len:] = 1
        
                yield {
                    "sequence": np.array(sequence)[:-1],
                    "target": np.array(sequence)[1:],
                    "mask": mask,
                    "question_len": question_len,
                    "solution_len": solution_len,
                }
            elif self.sequence_type == "question_only_decode":
                sequence = sequence + [3]
                mask = np.ones(len(soln_list_repr))
                yield {
                    "sequence": np.array(sequence),
                    "target": np.array(soln_list_repr),
                    "mask": mask,
                    "question_len": question_len,
                    "solution_len": solution_len,
                }
            elif self.sequence_type == "question_only":
                question_idx = answer_idx = 0
                if (
                    len(soln_list_repr) > 1
                    and sample_rng.rand() < self.p_curriculum
                ):
                    sampled_idx = sample_rng.randint(1, len(soln_list_repr))
                    if self.reverse_curriculum:
                        question_idx = sampled_idx
                    else:
                        answer_idx = sampled_idx

                sequence = sequence + [3] + soln_list_repr[:question_idx]
                soln_list_repr = [3] + soln_list_repr[:len(soln_list_repr) - answer_idx]

                sequence = sequence + [self.eos_token_id] * (self.context_len - len(sequence) + 1)
                soln_list_repr = soln_list_repr + [self.eos_token_id] * (self.context_len - len(soln_list_repr) + 1)

                # XXX: This disallows RL to update provided ground-truth tokens
                # question_len += question_idx

                mask = np.zeros(len(sequence))
                mask[question_len:] = 1
                
                yield {
                    "sequence": np.array(sequence),
                    "target": np.array(soln_list_repr),
                    "mask": mask,
                    "pointer_correct": question_idx + 1,
                    "question_len": question_len,
                    "solution_len": solution_len + 1,
                }
