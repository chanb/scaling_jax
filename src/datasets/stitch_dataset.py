import _pickle as pickle
import numpy as np

from gymnasium import spaces
from torch.utils.data import IterableDataset
from typing import Any, NamedTuple


class DataInfo(NamedTuple):
    data_path: str
    env_params: Any
    task_ids: list[int]
    num_tasks: int
    max_len: int
    buffer: Any


class GymnaxStitchDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_paths: list[str],
        seq_len: int,
        seed: int,
    ):
        self.seq_len = seq_len
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
                        max_len=data["data"]["reward"].shape[-1] - seq_len - 1,
                        buffer=data["data"],
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

            """
            1. Sample good performance episode
            2. Sample transitions from learning history
            3. Shuffle (1) into (2) <-- This should enforce stitching
            4. Append (1)
            """
            data_path_id = self._rng.randint(self.num_data_paths)
            data_info = self.data_infos[data_path_id]
            task_id = self._rng.randint(data_info.num_tasks)
            start_idx = self._rng.randint(data_info.max_len)
            buffer = data_info.buffer

            states = buffer["obs"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            actions = buffer["action"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            rewards = buffer["reward"][task_id][
                start_idx : start_idx + self.seq_len
            ]

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }
