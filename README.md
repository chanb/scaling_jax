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

### Downloading XLand-100B Datasets
See [here](https://github.com/dunnolab/xland-minigrid-datasets/tree/main) for detailed instructions:
```
curl -L -o xland-trivial-20b.hdf5 https://tinyurl.com/trivial-10k
```

## Example
```
CUDA_VISIBLE_DEVICES="0,1,2,3" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python src/main.py --config_path=configs/xland_ad.json
```


## Alliance CAN
Setup
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.6

pip install --upgrade pip --no-index
pip install jax --no-index
pip install optax flax --no-index
pip install chex dill matplotlib tensorboard seaborn tqdm --no-index
pip install gymnasium --no-index
pip install torch torchvision --no-index
pip install scikit-learn --no-index
pip install prefetch_generator --no-index
pip install h5py --no-index

pip install xminigrid~=0.8.0
```