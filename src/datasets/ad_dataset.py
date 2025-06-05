import _pickle as pickle
import h5py
import numpy as np

from gymnasium import spaces
from torch.utils.data import IterableDataset
from xminigrid.core.constants import NUM_ACTIONS, NUM_COLORS


class XMiniGridADDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
    ):
        self.data_file = None
        self.seq_len = seq_len
        self.data_path = data_path
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        with h5py.File(data_path, "r") as df:
            self.benchmark_id = df["0"].attrs["benchmark-id"]
            self.env_id = df["0"].attrs["env-id"]
            self.task_ids = list(df.keys())

            self.num_tasks = len(list(df.keys()))
            self.hists_per_task = df["0/rewards"].shape[0]
            self.max_len = df["0/rewards"].shape[-1] - seq_len - 1 # Exclude very last sample per history

            self.ruleset_ids = []

            for i in df.keys():
                self.ruleset_ids.append(df[i].attrs["ruleset-id"])

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

    def open_hdf5(self):
        self.data_file = h5py.File(self.data_path, "r")

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(self):
        if self.data_file is None:
            self.open_hdf5()
        
        while True:
            task_id = self._rng.choice(self.task_ids)
            learning_history_idx = self._rng.randint(self.hists_per_task)
            start_idx = self._rng.randint(self.max_len)

            states = self.decompress_obs(
                self.data_file[task_id]["states"][learning_history_idx][
                    start_idx : start_idx + self.seq_len
                ]
            )
            actions = self.data_file[task_id]["actions"][learning_history_idx][
                start_idx : start_idx + self.seq_len
            ]
            rewards = self.data_file[task_id]["rewards"][learning_history_idx][
                start_idx : start_idx + self.seq_len
            ]

            yield {
                "state": states, # (seq_len, 5, 5, 2)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }


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
            self.env_params = data["env_params"]
            self.buffer = data["data"]
            self.task_ids = np.arange(len(self.env_params))
            self.num_arms = self.env_params.shape[-1]
            self.num_tasks = len(self.task_ids)
            self.hists_per_task = 1

            # Exclude very last sample per history
            self.max_len = self.buffer["reward"].shape[-1] - seq_len - 1

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
            actions = self.buffer["actions"][task_id][
                start_idx : start_idx + self.seq_len
            ]
            rewards = self.buffer["rewards"][task_id][
                start_idx : start_idx + self.seq_len
            ]

            yield {
                "state": states, # (seq_len, 5, 5, 2)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }
