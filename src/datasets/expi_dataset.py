import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import _pickle as pickle
import numpy as np

from torch.utils.data import IterableDataset

from src.datasets.utils import DataInfo


class GymnaxExPIDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_paths: list[str],
        seq_len: int,
        skip_ep: int,
        seed: int,
    ):
        self.seq_len = seq_len
        self.skip_ep = skip_ep
        self.data_paths = data_paths
        self.num_data_paths = len(data_paths)
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.data_infos = []
        self.num_total_tasks = 0

        for path_i, data_path in enumerate(self.data_paths):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                if path_i == 0:
                    self._observation_space = data["observation_space"]
                    self._action_space = data["action_space"]

                self.data_infos.append(
                    DataInfo(
                        data_path=data_path,
                        env_params=data["env_params"],
                        task_ids=self.num_total_tasks + np.arange(len(data["env_params"])),
                        num_tasks=len(data["env_params"]),
                        max_len=data["learning_histories"]["reward"].shape[-1] - seq_len - 1,
                        buffer=data["learning_histories"],
                    )
                )
            self.num_total_tasks += self.data_infos[-1].num_tasks

        print("Loaded dataset")

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(self):
        while True:
            data_path_id = self._rng.randint(self.num_data_paths)
            data_info = self.data_infos[data_path_id]
            task_id = self._rng.randint(data_info.num_tasks)
            start_idx = self._rng.randint(data_info.max_len)
            buffer = data_info.buffer

            start_idxes = np.concatenate((
                [0],
                np.where(
                    buffer["done"][task_id] == 1
                )[0] + 1,
            ))

            curr_ep = self._rng.randint(len(start_idxes) - self.skip_ep)
            start_idx = self._rng.randint(
                start_idxes[curr_ep],
                start_idxes[curr_ep + 1],
            )

            all_idxes = np.arange(start_idx, start_idxes[curr_ep + 1])

            while len(all_idxes) < self.seq_len:
                curr_ep += self.skip_ep
                if curr_ep >= len(start_idxes) - 1:
                    curr_ep -= self.skip_ep
                all_idxes = np.concatenate((
                    all_idxes,
                    np.arange(start_idxes[curr_ep], start_idxes[curr_ep + 1]),
                ))

            remainder = len(all_idxes) % self.seq_len
            if remainder > 0:
                all_idxes = all_idxes[:-remainder]

            states = buffer["obs"][task_id][all_idxes]
            actions = buffer["action"][task_id][all_idxes]
            rewards = buffer["reward"][task_id][all_idxes]

            if np.any(np.isnan(rewards)) or np.any(np.isnan(actions)):
                continue

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
                "mask": np.ones_like(actions, dtype=np.float32),  # Mask for the sequence
            }
