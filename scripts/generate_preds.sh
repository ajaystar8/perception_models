#!/bin/bash
#SBATCH --job-name=generate_preds
#SBATCH --output=./sbatch_logs/generate_preds/finetuned_egoblind_blind_aware_%A.out
#SBATCH --error=./sbatch_logs/generate_preds/finetuned_egoblind_blind_aware_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=jiang
#SBATCH --gres=gpu:a6000:1

source /home/${USER}/.bashrc
source activate perception_models

MODEL_CKPT="/projects/torresani-lab/ajay/perception_models/checkpoints/finetune_egoblind/checkpoints/0000000500/"
EXPERIMENT_NAME="finetuned_plm8b_preds_blind_aware_prompt"
PROMPT_TYPE="blind_aware"

python /projects/torresani-lab/ajay/perception_models/generate_preds.py \
    --model_ckpt $MODEL_CKPT \
    --experiment_name $EXPERIMENT_NAME \
    --prompt_type $PROMPT_TYPE \
    --egoblind_format \
