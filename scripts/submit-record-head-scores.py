import os
import subprocess
import time
import numpy as np
from datetime import date
from monkey_patch_head_types import *

debug = False
command = subprocess.run if not debug else print

# List of models to evaluate. You can modify this list to include the models you want to evaluate.
# pretrained_model_name_or_paths = ['Qwen/Qwen2.5-7B', 'meta-llama/Llama-3.1-8B', 'allenai/OLMo-2-1124-7B', 'meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',\
#                                   'allenai/OLMo-2-1124-7B-SFT', 'allenai/OLMo-2-1124-7B-DPO', 'allenai/OLMo-2-1124-7B-Instruct',\
#                                   'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B',  'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B']
pretrained_model_name_or_paths = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-3B'] 
# 'allenai/OLMo-2-1124-7B-Instruct' didn't work due to data type issue causing NaNs/infs in head scores

# In order to save the percents file for each task, we run the evaluation script for each task separately
tasks, num_fewshot = 'mmlu', 5
# tasks, num_fewshot = 'winogrande', 5
# tasks, num_fewshot = 'piqa', 0

# Result dir prefix. This is where results will be saved. 
results_dir = f"results-record-head-scores"

slurm_script = 'scripts/lm-eval-record-head-scores.sh'

# Create head class (with dummy values) to access the generate_all_thresholds method
# already done: RandomHeads(0.0)
# track_head_classes = [HonorHeads(0.0), DormantHeads(0.0), \
#                       NormalizedDormantHeads(0.0), UnnormalizedHonorHeads(0.0), \
#                       EntropyHeads(0.0), NormalizedEntropyHeads(0.0)]

# track_head_classes += [ValueVectorMagnitudeFirstToken(0.0), ValueVectorMagnitudeNormalizedFirstToken(0.0), \
#                       ValueVectorAvgMagnitude(0.0), ValueVectorAvgNormalizedMagnitude(0.0), \
#                       HeadOutputMagnitudeLastToken(0.0), HeadOutputMagnitudeNormalizedLastToken(0.0), \
#                       HeadOutputMagnitudeNormalizedHeadLastToken(0.0)]
track_head_classes = [FullHeadOutput(0.0), FullHeadOutputNormalized(0.0)]

# Helper function to not submit jobs that have already been computed
def is_job_complete(pretrained_model_name_or_path, task, results_dir, track_head_type):
    results_file = f"{results_dir}/{pretrained_model_name_or_path.replace('/', '_')}/{task}/head_scores/{track_head_type}.pt"
    return os.path.exists(results_file)

count = 0
for task in tasks.split(','):
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        for track_head_type_class in track_head_classes:
            track_head_argname = track_head_type_class.argname
            if is_job_complete(pretrained_model_name_or_path, task, results_dir, track_head_argname):
                print(f"Job already complete: {pretrained_model_name_or_path}, {task}, {track_head_argname}")
                continue

            print(f'=> {slurm_script}\n\t{pretrained_model_name_or_path=}\n\t{task=}\n\t{track_head_argname=}\n\t{num_fewshot=}')
            command(['sbatch',
                    slurm_script,
                    pretrained_model_name_or_path, # 1st arg
                    task,                          # 2nd arg
                    results_dir,                   # 3rd arg
                    track_head_argname,               # 4th arg: track_head_type
                    str(num_fewshot)])             # 5th arg
            count += 1
            time.sleep(2) # To avoid overloading the scheduler

            if count >= 500:
                print('Reached 500 job submission limit, stopping here for now. Waiting 5 hours')
                exit(0)
print(f'Number of jobs submitted: {count}')