#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=0-01:30:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-2
#SBATCH --output=/home/chanb/scratch/logs/icrl_baselines/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source /home/chanb/research/venvs/aaai_expi/bin/activate

export REPO_PATH=/home/chanb/research/aaai_2026/scaling_jax

`sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/chanb/research/aaai_2026/scaling_jax/cc_scripts/baselines.dat`
echo ${SLURM_ARRAY_TASK_ID}
echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"
echo ${config_name}
echo "Starting run at: `date`"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python ${REPO_PATH}/src/main.py --config_path=${REPO_PATH}/configs/${config_name}

echo "Program test finished with exit code $? at: `date`"
