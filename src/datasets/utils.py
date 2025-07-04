from typing import Any, NamedTuple

class DataInfo(NamedTuple):
    data_path: str
    env_params: Any
    task_ids: list[int]
    num_tasks: int
    max_len: int
    buffer: Any
    expert_data: Any = None
