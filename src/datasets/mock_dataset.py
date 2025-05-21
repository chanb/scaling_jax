import numpy as np

from gymnasium import spaces
from torch.utils.data import IterableDataset
from xminigrid.core.constants import NUM_ACTIONS, NUM_COLORS, NUM_TILES


class MockXMiniGridADDataset(IterableDataset):
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

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(5, 5, 2), dtype=int)

    @property
    def action_space(self):
        return spaces.Discrete(NUM_ACTIONS)

    @staticmethod
    def decompress_obs(obs: np.ndarray) -> np.ndarray:
        return np.stack(np.divmod(obs, NUM_COLORS), axis=-1)

    def __iter__(self):
        return iter(self.get_sequences())

    def get_sequences(self):
        while True:
            states = np.concatenate((
                np.random.randint(0, NUM_TILES + 1, size=(self.seq_len, 5, 5, 1)),
                np.random.randint(0, NUM_COLORS, size=(self.seq_len, 5, 5, 1)),
            ), axis=-1)
            actions = np.random.randint(0, 6, size=(self.seq_len,))
            rewards = np.random.randn(self.seq_len)

            yield {
                "state": states, # (seq_len, 5, 5, 2)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
            }
