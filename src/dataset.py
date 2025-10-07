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

from src.utils import EmptyDatasetError, parse_dict


def get_iter(data_loader, data_sharding, dtype, stop_on_empty: bool=False):
    """
    Converts a DataLoader to an iterator that handles data sharding and dtype conversion.
    """

    loader = iter(data_loader)
    while True:
        try:
            batch = next(loader)
        except StopIteration:
            if stop_on_empty:
                raise EmptyDatasetError()
            else:
                loader = iter(data_loader)
                batch = next(loader)

        for k, v in batch.items():
            if hasattr(v, "numpy"):
                batch[k] = v.numpy()
            if np.issubdtype(batch[k].dtype, np.floating):
                batch[k] = batch[k].astype(dtype)
        yield batch


def get_dataset(config: SimpleNamespace, data_sharding, dtype, seed) -> Any:
    dataset_name = config.dataset_name
    dataset_kwargs = config.dataset_kwargs

    if dataset_name == "curriculum":
        from src.datasets.curriculum import Curriculum
        datasets = [
            get_dataset(
                parse_dict(config),
                data_sharding,
                dtype,
                seed,
            ) for config in dataset_kwargs.datasets
        ]
        dataset = Curriculum(
            datasets,
            dataset_kwargs.curriculum_schedule,
            dataset_kwargs.curriculum_type,
            seed,
        )
    elif dataset_name == "parity":
        from src.datasets.decision_problems import Parity
        dataset = Parity(
            dataset_kwargs.context_len,
            dataset_kwargs.max_level,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.train_val_ratio,
            getattr(dataset_kwargs, "exact", False),
            getattr(dataset_kwargs, "num_cot_tokens", 0),
            getattr(dataset_kwargs, "repeat", True),
            getattr(dataset_kwargs, "predict_eos", True),
        )
    else:
        raise NotImplementedError
    
    return dataset


def get_data_loader(config: SimpleNamespace, data_sharding, dtype) -> Any:
    """
    Returns a DataLoader for the specified dataset based on the configuration.
    """

    num_workers = getattr(config, "num_workers", 0)

    batch_size = config.batch_size
    dataset = get_dataset(
        config,
        data_sharding=data_sharding,
        dtype=dtype,
        seed=config.seeds.data_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    loader = get_iter(
        loader,
        data_sharding,
        dtype,
        stop_on_empty=not getattr(config, "repeat", True),
    )
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
