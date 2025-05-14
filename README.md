# Scaling Jax Experiments

## Installation
Python 3.10
```
pip install jax flax optax
```

## Example
```
CUDA_VISIBLE_DEVICES="0,1,2,3" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python src/main.py --config_path=local_utils/templates/ad.json
```
