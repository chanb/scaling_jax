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


class GymnaxStitchDataset(IterableDataset):
    """
    Data is collected using rejax.
    """

    def __init__(
        self,
        data_paths: list[str],
        seq_len: int,
        seed: int,
        all_token_pred: bool = False,
    ):
        self.seq_len = seq_len
        self.data_paths = data_paths
        self.num_data_paths = len(data_paths)
        self.seed = seed
        self.all_token_pred = all_token_pred
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
                        expert_data=data["expert_data"],
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
        if self.all_token_pred:
            def generate_mask(actions, expert_ep_len):
                return np.ones_like(actions, dtype=np.float32)
        else:
            def generate_mask(actions, expert_ep_len):
                mask = np.zeros_like(actions, dtype=np.float32)
                mask[-expert_ep_len:] = 1.0
                return mask
        while True:

            """
            TODO:
            Include next state
            """
            data_path_id = self._rng.randint(self.num_data_paths)
            data_info = self.data_infos[data_path_id]
            task_id = self._rng.randint(data_info.num_tasks)
            buffer = data_info.buffer
            expert_data = data_info.expert_data

            transition_idxes = self._rng.randint(
                data_info.max_len, size=(self.seq_len,)
            )
            states = buffer["obs"][task_id][
                transition_idxes
            ]

            actions = buffer["action"][task_id][
                transition_idxes
            ]

            rewards = buffer["reward"][task_id][
                transition_idxes
            ]

            # Fill expert data to the end
            expert_ep = self._rng.randint(
                expert_data["obss"][task_id].shape[0]
            )
            expert_ep_len = np.where(expert_data["dones"][task_id, expert_ep] == 1)[0][0] + 1
            states[-expert_ep_len:] = expert_data["obss"][task_id, expert_ep, :expert_ep_len]
            actions[-expert_ep_len:] = expert_data["actions"][task_id, expert_ep, :expert_ep_len]
            rewards[-expert_ep_len:] = expert_data["rewards"][task_id, expert_ep, :expert_ep_len]

            # Replace some transitions with expert data to encourage stitching
            replacement_idxes = self._rng.permutation(
                np.arange(self.seq_len - expert_ep_len)
            )[:expert_ep_len]
            states[replacement_idxes] = expert_data["obss"][task_id, expert_ep, :expert_ep_len]
            actions[replacement_idxes] = expert_data["actions"][task_id, expert_ep, :expert_ep_len]
            rewards[replacement_idxes] = expert_data["rewards"][task_id, expert_ep, :expert_ep_len]

            if np.any(np.isnan(rewards)) or np.any(np.isnan(actions)):
                continue

            # Only care about the last expert episode
            mask = generate_mask(actions, expert_ep_len)

            yield {
                "state": states, # (seq_len,)
                "action": actions, # (seq_len,)
                "reward": rewards, # (seq_len,)
                "target": actions, # (seq_len,)
                "mask": mask,
            }
