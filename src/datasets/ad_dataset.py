import _pickle as pickle
import numpy as np

from gymnasium import spaces
from torch.utils.data import IterableDataset
from typing import Any, NamedTuple


class BanditADDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
    ):
        self.seq_len = seq_len
        self.data_path = data_path
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.env_params = np.array(data["env_params"])
            self.buffer = data["data"]
            self.task_ids = np.arange(len(self.env_params))
            self.num_arms = self.env_params.shape[-1]
            self.num_tasks = len(self.task_ids)
            self.hists_per_task = 1

            # Exclude very last sample per history
            self.max_len = self.buffer["reward"].shape[-1] - seq_len - 1
        print("Loaded dataset")

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def action_space(self):
        return spaces.Discrete(self.num_arms)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(self):
        while True:
            task_id = self._rng.choice(self.task_ids)
            start_idx = self._rng.randint(self.max_len)

            states = self.buffer["obs"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            actions = self.buffer["action"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            rewards = self.buffer["reward"][task_id][
                start_idx : start_idx + self.seq_len
            ]

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }


class NonStationaryBanditADDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
    ):
        self.seq_len = seq_len
        self.data_path = data_path
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.env_params = np.array(data["env_params"])
            self.buffer = data["data"]
            self.task_ids = np.arange(len(self.env_params))
            self.num_arms = self.env_params.shape[-1]
            self.num_tasks = len(self.task_ids)
            self.hists_per_task = 1

            # Exclude very last sample per history
            self.max_len = self.buffer["reward"].shape[-1]
        print("Loaded dataset")

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def action_space(self):
        return spaces.Discrete(self.num_arms)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(self):
        while True:
            task_id = self._rng.choice(self.task_ids)
            start_idx = self._rng.randint(self.max_len)

            states = self.buffer["obs"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            actions = self.buffer["action"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            rewards = self.buffer["reward"][task_id][
                start_idx : start_idx + self.seq_len
            ]

            remainder = self.seq_len - len(states)
            if remainder > 0:
                new_task_id = self._rng.choice(self.task_ids)
                states = np.concatenate((states, self.buffer["obs"][new_task_id][
                    :remainder
                ]))

                actions = np.concatenate((self.buffer["action"][new_task_id][
                    :remainder
                ]))

                rewards = np.concatenate((self.buffer["reward"][new_task_id][
                    :remainder
                ]))

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }


class DataInfo(NamedTuple):
    data_path: str
    env_params: Any
    task_ids: list[int]
    num_tasks: int
    max_len: int
    buffer: Any


class GymnaxADDataset(IterableDataset):
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
            self.num_total_tasks += self.data_infos[data_path].num_tasks

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
