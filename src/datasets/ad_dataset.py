import h5py
import numpy as np

from gymnasium import spaces
from torch.utils.data import Dataset
from xminigrid.core.constants import NUM_ACTIONS, NUM_COLORS


class XMiniGridADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
    ):
        self.data_file = None
        self.seq_len = seq_len
        self.data_path = data_path

        with h5py.File(data_path, "r") as df:
            self.benchmark_id = df["0"].attrs["benchmark-id"]
            self.env_id = df["0"].attrs["env-id"]

            self.num_tasks = len(list(df.keys()))
            self.hists_per_task = df["0/rewards"].shape[0]
            max_len = df["0/rewards"].shape[-1] - seq_len

            self.segment_len = seq_len // 2
            self.num_segments = len(range(1, max_len, self.segment_len))
            self.dataset_len = self.num_tasks * self.hists_per_task * self.num_segments
            self.ruleset_ids = []

            for i in df.keys():
                self.ruleset_ids.append(df[i].attrs["ruleset-id"])

    def __get_idxs(self, idx):
        task_idx, other_idx = divmod(idx, self.hists_per_task * self.num_segments)
        hist_idx, segment_idx = divmod(other_idx, self.num_segments)
        start_idx = 1 + segment_idx * self.segment_len

        return str(task_idx), hist_idx, start_idx

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(5, 5, 2), dtype=int)

    @property
    def action_space(self):
        return spaces.Discrete(NUM_ACTIONS)

    @property
    def trajectories_metadata(self):
        return self.benchmark_id, self.env_id, self.ruleset_ids

    @staticmethod
    def decompress_obs(obs: np.ndarray) -> np.ndarray:
        return np.stack(np.divmod(obs, NUM_COLORS), axis=-1)

    def __len__(self):
        return self.dataset_len

    def open_hdf5(self):
        self.data_file = h5py.File(self.data_path, "r")

    def __getitem__(self, idx):
        if self.data_file is None:
            self.open_hdf5()

        task_idx, learning_history_idx, start_idx = self.__get_idxs(idx)

        states = self.decompress_obs(
            self.data_file[task_idx]["states"][learning_history_idx][
                start_idx : start_idx + self.seq_len
            ]
        )
        actions = self.data_file[task_idx]["actions"][learning_history_idx][
            start_idx : start_idx + self.seq_len
        ]
        rewards = self.data_file[task_idx]["rewards"][learning_history_idx][
            start_idx : start_idx + self.seq_len
        ]
        dones = self.data_file[task_idx]["dones"][learning_history_idx][
            start_idx : start_idx + self.seq_len
        ]

        return {
            "state": states, # (seq_len, 5, 5, 2)
            "action": actions, # (seq_len,)
            "reward": rewards, # (seq_len,)
            "done": dones, # (seq_len,)
            "target": actions, # (seq_len,)
        }
