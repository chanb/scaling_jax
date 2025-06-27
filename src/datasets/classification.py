import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces
from torch.utils.data import IterableDataset

import numpy as np
import timeit


class Classification(IterableDataset):
    def __init__(
        self,
        context_len: int,
        num_high_prob_classes: int,
        num_low_prob_classes: int,
        p_high: float,
        p_relevant_context: float,
        num_dims: int,
        seed: int,
        train: bool,
        query_cond: str = "none",
        input_noise_std: float = 0.0,
        label_noise: float = 0.0,
        num_relevant_contexts: int = None,
        target_in_context: bool = False,
        flip_label: bool = False,
    ):
        assert 0.0 < p_high < 1.0
        assert p_high / num_high_prob_classes >= (1 - p_high) / num_low_prob_classes
        assert num_relevant_contexts is None or num_relevant_contexts > 0

        self.num_high_prob_classes = num_high_prob_classes
        self.num_low_prob_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.p_high = p_high
        self.p_relevant_context = p_relevant_context
        self.low_prob = 1 - p_high
        self.num_dims = num_dims
        self.train = train
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.label_noise = label_noise
        self.context_len = context_len
        self.num_relevant_contexts = num_relevant_contexts
        self.target_in_context = target_in_context
        self.query_cond = query_cond
        self.flip_label = flip_label
        self.rng = np.random.RandomState(seed)

        self.centers = self.rng.standard_normal(size=(self.num_classes, self.num_dims))
        self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Discrete(self.num_classes)

    def __iter__(self):
        return iter(self.get_sequences())

    def generate_sample(self, rng, weights):
        targets = np.full(
            shape=(self.context_len,),
            fill_value=rng.choice(len(weights), p=weights),
        )

        if self.query_cond == "high_prob":
            targets[-1] = rng.choice(
                self.num_high_prob_classes,
            )
        elif self.query_cond == "low_prob":
            targets[-1] = (
                rng.choice(self.num_low_prob_classes,)
                + self.num_high_prob_classes
            )

        relevant_context_mask = rng.uniform() < self.p_relevant_context

        if relevant_context_mask:
            num_relevant_contexts = (
                rng.randint(self.context_len - 1)
                if self.num_relevant_contexts is None else
                self.num_relevant_contexts
            )
        else:
            num_relevant_contexts = 0
        
        query_context_identical = np.sum(
            targets[:-1] == targets[[-1]],
            axis=-1,
        )
        
        if num_relevant_contexts < self.context_len - 1:
            while query_context_identical != num_relevant_contexts:
                targets[num_relevant_contexts:-1] = rng.choice(
                    self.num_classes,
                    size=(
                        self.context_len - 1 - num_relevant_contexts,
                    ),
                    p=weights,
                )

                query_context_identical = np.sum(
                    targets[:-1] == targets[[-1]],
                    axis=-1,
                )

            targets[:-1] = np.random.default_rng(self.seed).permuted(targets[:-1])

        examples = self.centers[targets]
        examples += self.input_noise_std * rng.randn(*examples.shape)
        return examples, targets, num_relevant_contexts

    def get_sequences(
        self,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.p_high / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        if self.train:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random.RandomState(self.seed + 1)

        while True:
            examples, targets, num_relevant_contexts = self.generate_sample(rng, weights)

            # OOD labels: Make sure OOD label is still within the same frequency class
            if self.flip_label:
                high_prob_class_idxes = np.where(targets < self.num_high_prob_classes)[0]
                low_prob_class_idxes = np.where(targets >= self.num_high_prob_classes)[0]
                targets[high_prob_class_idxes] = (
                    targets[high_prob_class_idxes] + 1
                ) % self.num_high_prob_classes
                targets[low_prob_class_idxes] = (
                    targets[low_prob_class_idxes] - self.num_high_prob_classes + 1
                ) % self.num_low_prob_classes + self.num_high_prob_classes

            if rng.uniform() < self.label_noise:
                # All labels are randomly sampled such that they're not the original label
                new_targets = rng.choice(
                    self.num_classes - 1,
                    size=(self.num_classes,),
                )[targets]
                new_targets[new_targets >= targets] += 1
                new_targets = new_targets % self.num_classes

                # With some probability, change the query label to a different class
                # This is to ensure that P*(target | query) == P*(target | query, context)
                if (
                    num_relevant_contexts > 0
                    and not self.target_in_context
                    and rng.uniform() < self.label_noise
                ):
                    while (
                        new_targets[-1] == targets[-1]
                        or new_targets[-1] == targets[np.where(targets[:-1] == targets[-1])[0][0]]
                    ):
                        new_targets[-1] = rng.choice(self.num_classes)

                targets = new_targets

            one_hot = np.zeros((self.context_len, self.num_classes))
            one_hot[np.arange(self.context_len), targets] = 1

            yield {
                "example": examples,
                "target": one_hot,
                "mask": np.eye(self.context_len, dtype=np.float32)[-1],
            }
