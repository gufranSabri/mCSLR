#!/bin/bash
#SBATCH --job-name=mcslr
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=g202302610@kfupm.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=12:00:00

module load python/3.11.5 cuda/12.2 gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

bash ./scripts/install.sh

# Timestamp for unique work directory
timestamp=$(date +"%d_%H_%M_%S")
work_dir="./work_dir/v2__${timestamp}"

export PYTHONDONTWRITEBYTECODE=1
python main.py --work-dir "$work_dir" --config ./configs/baseline.yaml --slurm-mode
python main.py --work-dir "$work_dir" --config "$work_dir/baseline.yaml"  --mode test --slurm-mode