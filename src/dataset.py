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

import jax

from src.datasets.ad_dataset import XMiniGridADataset


def get_iter(data_loader):
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
        yield batch


def get_data_loader(config: SimpleNamespace) -> Any:
    dataset_name = config.dataset_name
    dataset_kwargs = config.dataset_kwargs

    num_workers = getattr(config, "num_workers", 0)
    if dataset_name == "xland_ad":
        batch_size = config.batch_size
        shuffle = True
        drop_last = True
        dataset = XMiniGridADataset(
            dataset_kwargs.data_path,
            dataset_kwargs.seq_len,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    elif dataset_name == "xland_dpt":
        pass
    else:
        raise NotImplementedError

    loader = get_iter(loader)
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
