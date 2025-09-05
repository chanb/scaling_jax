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

from src.utils import parse_dict


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
    elif dataset_name == "linear_regression":
        from src.datasets.linear_regression import ICLinearRegression
        dataset = ICLinearRegression(
            dataset_kwargs.num_tasks,
            dataset_kwargs.num_dims,
            dataset_kwargs.context_len,
            seed,
            dataset_kwargs.train,
            dataset_kwargs.input_noise_std,
            dataset_kwargs.label_noise_std,
            dataset_kwargs.sparsity,
            dataset_kwargs.target_generator,
        )
    elif dataset_name == "classification":
        from src.datasets.classification import Classification
        dataset = Classification(
            dataset_kwargs.context_len,
            dataset_kwargs.num_high_prob_classes,
            dataset_kwargs.num_low_prob_classes,
            dataset_kwargs.p_high,
            dataset_kwargs.p_relevant_context,
            dataset_kwargs.num_dims,
            seed,
            dataset_kwargs.train,
            dataset_kwargs.query_cond,
            dataset_kwargs.input_noise_std,
            dataset_kwargs.label_noise,
            dataset_kwargs.num_relevant_contexts,
            dataset_kwargs.target_in_context,
            dataset_kwargs.flip_label,
        )
    elif dataset_name == "uniform_classification":
        from src.datasets.classification import UniformClassification
        dataset = UniformClassification(
            dataset_kwargs.context_len,
            dataset_kwargs.num_classes,
            dataset_kwargs.p_relevant_context,
            dataset_kwargs.num_dims,
            seed,
            dataset_kwargs.train,
            dataset_kwargs.query_cond,
            dataset_kwargs.input_noise_std,
            dataset_kwargs.label_noise,
            dataset_kwargs.num_relevant_contexts,
            dataset_kwargs.target_in_context,
            dataset_kwargs.flip_label,
        )
    elif dataset_name == "nary_strings":
        from src.datasets.nary_strings import NaryStrings
        dataset = NaryStrings(
            dataset_kwargs.n_ary,
            dataset_kwargs.num_levels,
            dataset_kwargs.context_len,
            dataset_kwargs.min_sequence_len,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
        )
    elif dataset_name == "k_parity":
        from src.datasets.k_parity import KParity
        dataset = KParity(
            dataset_kwargs.sequence_length,
            dataset_kwargs.k,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
            dataset_kwargs.train_val_ratio,
        )
    elif dataset_name == "threshold_sum":
        from src.datasets.sum import ThresholdSum
        dataset = ThresholdSum(
            dataset_kwargs.context_len,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
            dataset_kwargs.train_val_ratio,
            dataset_kwargs.include_boundary,
        )
    elif dataset_name == "addition":
        from src.datasets.sum import Addition
        dataset = Addition(
            dataset_kwargs.context_len,
            dataset_kwargs.max_int,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
            dataset_kwargs.train_val_ratio,
            getattr(dataset_kwargs, "right_to_left", False),
            getattr(dataset_kwargs, "shuffle", True),
        )
    elif dataset_name == "xor":
        from src.datasets.sum import XOR
        dataset = XOR(
            dataset_kwargs.context_len,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
            dataset_kwargs.train_val_ratio,
            dataset_kwargs.include_boundary,
        )
    elif dataset_name == "linear_system":
        from src.datasets.dynamical_systems import LinearSystem
        dataset = LinearSystem(
            dataset_kwargs.context_len,
            dataset_kwargs.num_dims,
            dataset_kwargs.train,
            seed,
            dataset_kwargs.sequence_type,
            dataset_kwargs.show_A,
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

    loader = get_iter(loader, data_sharding, dtype)
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
