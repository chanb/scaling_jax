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
import jax.numpy as jnp
import numpy as np

from src.datasets.ad_dataset import XMiniGridADDataset
from src.datasets.dpt_dataset import XMiniGridDPTDataset


def get_iter(data_loader, data_sharding, half_precision):
    if half_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

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
            if not np.isdtype(batch[k].dtype, "integral"):
                batch[k] = batch[k].astype(dtype)
        yield jax.device_put(batch, data_sharding)


def get_data_loader(config: SimpleNamespace, data_sharding) -> Any:
    dataset_name = config.dataset_name
    dataset_kwargs = config.dataset_kwargs

    num_workers = getattr(config, "num_workers", 0)

    if dataset_name.startswith("xland"):
        batch_size = config.batch_size
        if dataset_name == "xland_ad":
            dataset = XMiniGridADDataset(
                dataset_kwargs.data_path,
                dataset_kwargs.seq_len,
                config.seeds.data_seed,
            )
        elif dataset_name == "xland_dpt":
            dataset = XMiniGridDPTDataset(
                dataset_kwargs.data_path,
                dataset_kwargs.seq_len,
                config.seeds.data_seed,
            )
        else:
            raise NotImplementedError
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    loader = get_iter(loader, data_sharding, config.half_precision)
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
