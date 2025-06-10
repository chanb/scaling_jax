import _pickle as pickle
import numpy as np

from gymnasium import spaces
from torch.utils.data import IterableDataset


class BanditDPTDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        cut_off: int,
        seed: int,
    ):
        self.seq_len = seq_len
        self.data_path = data_path
        self.cut_off = cut_off
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.env_params = np.array(data["env_params"])
            self.buffer = data["data"]
            self.best_actions = np.array(np.argmax(self.env_params, axis=-1))
            self.task_ids = np.arange(len(self.env_params))
            self.num_arms = self.env_params.shape[-1]
            self.num_tasks = len(self.task_ids)
            self.hists_per_task = 1

            # Exclude very last sample per history
            self.max_len = min(self.buffer["reward"].shape[-1], self.cut_off)
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
            idxes = self._rng.randint(self.max_len, size=(self.seq_len,))

            states = self.buffer["obs"][task_id][idxes]
            actions = self.buffer["action"][task_id][idxes]
            rewards = self.buffer["reward"][task_id][idxes]

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": np.full_like(
                    actions,
                    fill_value=self.best_actions[task_id]
                ), # (seq_len,)
            }
