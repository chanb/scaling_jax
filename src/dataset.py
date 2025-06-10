import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.datasets.ad_dataset import BanditADDataset
from src.datasets.dpt_dataset import BanditDPTDataset


def get_iter(data_loader, data_sharding, dtype):
    """
    Converts a DataLoader to an iterator that handles data sharding and dtype conversion.
    """

    loader = iter(data_loader)
    while True:
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(data_loader)
            batch = next(loader)

        for k, v in batch.items():
            if hasattr(v, "numpy"):
                batch[k] = v.numpy()
            if np.issubdtype(batch[k].dtype, np.floating):
                batch[k] = batch[k].astype(dtype)
        yield batch


def get_data_loader(config: SimpleNamespace, data_sharding, dtype) -> Any:
    """
    Returns a DataLoader for the specified dataset based on the configuration.
    """

    dataset_name = config.dataset_name
    dataset_kwargs = config.dataset_kwargs

    num_workers = getattr(config, "num_workers", 0)

    batch_size = config.batch_size
    if dataset_name == "bandit_ad":
        dataset = BanditADDataset(
            dataset_kwargs.data_path,
            dataset_kwargs.seq_len,
            config.seeds.data_seed,
        )
    elif dataset_name == "bandit_dpt":
        dataset = BanditDPTDataset(
            dataset_kwargs.data_path,
            dataset_kwargs.seq_len,
            dataset_kwargs.cut_off,
            config.seeds.data_seed,
        )
    else:
        raise NotImplementedError

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    loader = get_iter(loader, data_sharding, dtype)
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
