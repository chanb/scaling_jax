import h5py
import numpy as np
import random
import xminigrid

from typing import List, Tuple
from gymnasium import spaces
from torch.utils.data import IterableDataset
from xminigrid.core.constants import NUM_ACTIONS, NUM_COLORS


class XMiniGridDPTDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
    ):
        super().__init__()

        self.data_file = None
        self.seq_len = seq_len
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        with h5py.File(data_path, "r") as df:
            self.num_tasks = len(df.keys())
            self.data_path = data_path
            self.indexes = []
            self.ruleset_ids = []
            
            self.benchmark_id = df["0"].attrs["benchmark-id"]
            self.env_id = df["0"].attrs["env-id"]
            self.max_len = df["0"]["states"][0].shape[0] - seq_len
            self.learning_history_len = self.max_len + seq_len
            self.num_learning_histories = df["0"]["states"].shape[0]
            self.task_ids = list(df.keys())

            for task_id in df.keys():
                self.ruleset_ids.append(df[task_id].attrs["ruleset-id"])
    
    def open_hdf5(self):
        self.data_file = h5py.File(self.data_path, "r")
    
    @staticmethod
    def get_episode_max_steps(env_id: str) -> int:
        env, env_params = xminigrid.make(env_id)
        return env_params.max_steps

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(5, 5, 2), dtype=int)

    @property
    def action_space(self):
        return spaces.Discrete(NUM_ACTIONS)
    
    @property
    def trajectories_metadata(self) -> Tuple[str, str, List[np.int64]]:
        return self.benchmark_id, self.env_id, self.ruleset_ids
    
    @staticmethod
    def decompress_obs(obs: np.ndarray) -> np.ndarray:
        return np.stack(np.divmod(obs, NUM_COLORS), axis=-1)
    
    def __iter__(self):
        return iter(self.get_sequences())
    
    def get_sequences(self):
        if self.data_file is None:
            self.open_hdf5()
        
        while True:
            task_id = self._rng.choice(self.task_ids)
            query_learning_history = self._rng.randint(0, self.num_learning_histories - 1)
            query_idx = self._rng.randint(0, self.max_len - 1)

            context_learning_history = self._rng.randint(0, self.num_learning_histories - 1)
            context_indexes = self._rng.randint(0, self.learning_history_len - 2, size=self.seq_len)

            query_states = self.decompress_obs(
                self.data_file[task_id]["states"][query_learning_history][[query_idx]]
            )
            states = self.decompress_obs(
                self.data_file[task_id]["states"][context_learning_history][context_indexes]
            )
            next_states = self.decompress_obs(
                self.data_file[task_id]["states"][context_learning_history][context_indexes + 1]
            )
            actions = self.data_file[task_id]["actions"][context_learning_history][context_indexes]
            target_actions = self.data_file[task_id]["expert_actions"][query_learning_history][query_idx]
            rewards = self.data_file[task_id]["rewards"][context_learning_history][context_indexes]

            yield {
                "query_state": query_states, # (1, 5, 5, 2)
                "state": states, # (seq_len, 5, 5, 2)
                "next_state": next_states, # (seq_len, 5, 5, 2)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": target_actions, # (seq_len,)
            }
