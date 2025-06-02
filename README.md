# Scaling Jax Experiments

## Installation
Python 3.10
```
pip install jax # or pip install jax[cuda12]
pip install flax==0.10.4
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
# GPU devices
CUDA_VISIBLE_DEVICES="0,1,2,3" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python src/main.py --config_path=configs/xland_ad.json

# CPU devices
xla_force_host_platform_device_count=8 python src/main.py --config_path=configs/xland_ad.json
```


## Alliance CAN
Setup
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

python -m venv <venv_name>
source <venv_name>/bin/activate

pip install --upgrade pip --no-index
pip install "jax[cuda12]" --no-index
pip install optax flax --no-index
pip install chex dill matplotlib tensorboard seaborn tqdm --no-index
pip install gymnasium --no-index
pip install torch torchvision --no-index
pip install scikit-learn --no-index
pip install prefetch_generator --no-index
pip install h5py --no-index

pip install xminigrid~=0.8.0
```

### Running:
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source <PATH_TO_VENV>/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python <PATH_TO_REPO>/src/main.py --config_path=<PATH_TO_REPO>/configs/xland_ad.json
```

### Interactive:
```
salloc --time=0:20:00 --mem=500GB --cpus-per-task=64 --gres=gpu:l40s:4 --account=aip-schuurma
```

Tensorboard:
```
Compute node $: tensorboard --logdir=. --host 0.0.0.0 --load_fast false

Local $: ssh -N -f -L localhost:6007:<node_name>:6006 <username>@vulcan.all
iancecan.ca
```