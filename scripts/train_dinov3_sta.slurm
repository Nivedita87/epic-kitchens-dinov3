#!/bin/bash
#SBATCH --job-name=epic-dinov3-sta
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
#SBATCH --output=/gpfs1/scratch/380/vsc38054/epic-dinov3/logs/dinov3-sta-%j.log

set -euo pipefail
if [[ ! -r scripts/train_dinov3_sta.py ]]; then
    echo "Missing scripts/train_dinov3_sta.py. Upload it to epic-work/scripts/." >&2
    exit 1
fi

module load cluster/wice/gpu_p100 Python/3.11.5-GCCcore-13.2.0
source /scratch/leuven/380/vsc38054/epic-dinov3-venv/bin/activate
export TORCH_HOME=/scratch/leuven/380/vsc38054/torch-cache
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run=/gpfs1/scratch/380/vsc38054/epic-dinov3
args=(
    --ar-checkpoint "$run/dinov3_base_colour_384_fullval_v1"
    --output "$run/dinov3_base_sta_384_5s_v1"
    --video-root /scratch/leuven/380/vsc38054/epic-kitchens/videos_640x360
)
python -u scripts/train_dinov3_sta.py --smoke "${args[@]}"
python -u scripts/train_dinov3_sta.py "${args[@]}"
