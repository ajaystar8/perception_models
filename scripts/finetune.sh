#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=scripts/sbatch_logs/finetune/plm8b_egoblind_%A.out
#SBATCH --error=scripts/sbatch_logs/finetune/plm8b_egoblind_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:a6000:8
#SBATCH --partition=jiang

# Get GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1  # Fallback to 1 if nvidia-smi fails
    GPU_TYPE="Unknown"
else
    # Get unique GPU types
    GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd ", ")
fi

echo "Launching testing with ${NUM_GPUS} GPU(s): ${GPU_TYPE}"

cd ${SLURM_SUBMIT_DIR}
source /home/${USER}/.bashrc
source activate perception_models

torchrun --nproc-per-node 8 -m apps.plm.train config=apps/plm/configs/finetune/plm_8b_custom.yaml 