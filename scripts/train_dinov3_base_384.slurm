#!/bin/bash
#SBATCH --job-name=epic-dinov3-base
#SBATCH --clusters=wice
#SBATCH --account=lp_edu_rdlab
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=44000M
#SBATCH --time=48:00:00
#SBATCH --chdir=/scratch/leuven/380/vsc38054/epic-work
#SBATCH --output=/scratch/leuven/380/vsc38054/epic-work/results/logs/dinov3-base-train-%j.log

set -euo pipefail
module load cluster/wice/gpu_p100 Python/3.11.5-GCCcore-13.2.0
source /scratch/leuven/380/vsc38054/epic-dinov3-venv/bin/activate
export TORCH_HOME=/scratch/leuven/380/vsc38054/torch-cache
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/train_dinov3_base_384.py
