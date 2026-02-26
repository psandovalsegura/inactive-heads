#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --job-name=eahd
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-%x.out
#--SBATCH --output=slurm-%j-%x.out
#--SBATCH --output=/dev/null
start_time=$(date +%s)

MODEL_PATH=$1
TASKS=$2
RESULTS_DIR=$3
TRACK_HEAD_TYPE=$4
NUM_FEWSHOT=$5

# A6000: batch_size=32
export TOKENIZERS_PARALLELISM=true
export NUMEXPR_MAX_THREADS=64

python evaluate_attention_heads_drop.py ${MODEL_PATH} ${TASKS} \
       --results_dir ${RESULTS_DIR} \
       --track_head_type ${TRACK_HEAD_TYPE} \
       --num_fewshot ${NUM_FEWSHOT} \
       --batch_size auto

# Log duration
end_time=$(date +%s)
duration=$((end_time - start_time))

days=$((duration / 86400))
hours=$(( (duration % 86400) / 3600 ))
minutes=$(( (duration % 3600) / 60 ))

echo "--------------------"
echo "Job took: ${days}d ${hours}h ${minutes}m"
echo "--------------------"