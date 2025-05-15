# Scaling Jax Experiments

## Installation
Python 3.10
```
pip install jax[cuda12] flax optax
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install chex optax dill gymnasium scikit-learn matplotlib seaborn tqdm tensorboard h5py
pip install prefetch_generator
pip install xminigrid~=0.8.0
```

## Example
```
CUDA_VISIBLE_DEVICES="0,1,2,3" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python src/main.py --config_path=configs/xland_ad.json
```
