import os
import re
import subprocess
import time
import numpy as np
from datetime import date
from monkey_patch_head_types import *

debug = False
verbose = False
command = subprocess.run if not debug else print

# List of models to evaluate. You can modify this list to include the models you want to evaluate.
# pretrained_model_name_or_paths = ['meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B-Instruct',\
#     'allenai/OLMo-2-1124-7B', 'allenai/OLMo-2-1124-7B-SFT', 'allenai/OLMo-2-1124-7B-DPO', 'allenai/OLMo-2-1124-7B-Instruct',\
#     'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B']
pretrained_model_name_or_paths = ['meta-llama/Llama-3.1-8B-Instruct']


# In order to save the percents file for each task, we run the evaluation script for each task separately
tasks, num_fewshot = 'mmlu', 5
# Other tasks included in the paper:
# tasks, num_fewshot = 'gsm8k', 5
# tasks, num_fewshot = 'winogrande', 5
# tasks, num_fewshot = 'piqa', 0

# Result dir prefix. This is where results will be saved. 
date_str = date.today().isoformat()
results_dir = f"results-{date_str}-attention-heads-drop"

# Both slurm scripts are identical except for GPU type, to spread the job load
slurm_script1 = 'scripts/lm-eval-attention-heads-drop.sh' # A6000
slurm_script2 = 'scripts/lm-eval-attention-heads-drop-1.sh' # L40S

# Create head class (with dummy threshold 0.0) to access the generate_all_thresholds method
# track_head_classes = [DormantHeads(0.0), HonorHeads(0.0), \
#                       NormalizedDormantHeads(0.0), UnnormalizedHonorHeads(0.0), \
#                       EntropyHeads(0.0), NormalizedEntropyHeads(0.0)]
# track_head_classes += [ValueVectorMagnitudeFirstToken(0.0), ValueVectorMagnitudeNormalizedFirstToken(0.0), \
#                       ValueVectorAvgMagnitude(0.0), ValueVectorAvgNormalizedMagnitude(0.0), \
#                       HeadOutputMagnitudeLastToken(0.0), HeadOutputMagnitudeNormalizedLastToken(0.0), \
#                       HeadOutputMagnitudeNormalizedHeadLastToken(0.0)]
# track_head_classes += [RandomHeads(0.0)]
# track_head_classes += [FullHeadOutput(0.0), FullHeadOutputNormalized(0.0)]
track_head_classes = [RandomHeads(0.0), DormantHeads(0.0), HonorHeads(0.0)]


# Helper function to not submit jobs that have already been computed
def is_job_complete(pretrained_model_name_or_path, task, results_dir, track_head_type):
    results_file = f"{results_dir}/{pretrained_model_name_or_path.replace('/', '_')}/{task}/{track_head_type}.json"
    return os.path.exists(results_file)

def is_job_running(pretrained_model_name_or_path, task, track_head_type):
    """
    Checks if a SLURM job with specific argparse parameters is currently running.

    This function queries SLURM for the user's running jobs, constructs the
    expected output filename for each, and reads the file to check for a
    matching configuration in the argparse Namespace.

    Args:
        pretrained_model_name_or_path (str): The model name to check for.
        task (str): The task name to check for (e.g., 'mmlu').
        track_head_type (str): The specific head tracking type to check for.

    Returns:
        bool: True if a matching job is found running, False otherwise.
    """
    # Get the current user's running jobs from SLURM
    user = os.getenv('USER')
    result = subprocess.run(
        ['squeue', '-u', user],
        capture_output=True,
        text=True,
        check=True
    )
    squeue_output = result.stdout

    # Skip the header line and process each job line
    job_lines = squeue_output.strip().split('\n')[1:]

    for line in job_lines:
        parts = line.split()
        if not parts:
            continue

        job_id = parts[0]
        job_name = parts[2]
        job_status = parts[4]

        # We only care about jobs that are currently Running ('R')
        if job_status != 'R':
            continue

        # Construct the expected SLURM output filename
        slurm_file = f"slurm-{job_id}-{job_name}.out"

        try:
            with open(slurm_file, 'r') as f:
                content = f.read()

                # Find the line containing the argparse Namespace
                match = re.search(r"Namespace\(.*\)", content)
                if not match:
                    continue
                
                namespace_str = match.group(0)

                # Define the patterns to look for
                model_check = f"pretrained_model_name_or_path='{pretrained_model_name_or_path}'"
                task_check = f"tasks=['{task}']" # Assumes a single task in a list
                head_type_check = f"track_head_type={track_head_type}"

                # If all three parameters match, we've found our job
                if (model_check in namespace_str and
                    task_check in namespace_str and
                    head_type_check in namespace_str):
                    if verbose: print(f"✅ Found matching running job: ID {job_id} ({slurm_file})")
                    if verbose: print(f"Job already running with {pretrained_model_name_or_path=}, {task=}, {track_head_type=}")
                    return True

        except FileNotFoundError:
            # This is expected if a job just started or has a different output setup.
            # We simply ignore it and check the next job.
            continue
        except Exception as e:
            print(f"⚠️ Could not read or parse file {slurm_file}: {e}")
            continue

    return False

count = 0
max_jobs = 500
total_count = 0
for task in tasks.split(','):
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        for track_head_type_class in track_head_classes:
            thresholds = track_head_type_class.generate_all_thresholds(pretrained_model_name_or_path, task)
            for threshold in thresholds:
                track_head_type_class.set_threshold(threshold)
                track_head_argname = track_head_type_class.argname
                if is_job_complete(pretrained_model_name_or_path, task, results_dir, track_head_argname):
                    if verbose: print(f"Job already complete: {pretrained_model_name_or_path}, {task}, {track_head_argname}")
                    continue

                if is_job_running(pretrained_model_name_or_path, task, track_head_argname):
                    continue
                
                slurm_script = slurm_script1 if count % 2 == 0 else slurm_script2 # Alternate between the two scripts
                # print(f'=> {slurm_script}\n\t{pretrained_model_name_or_path=}\n\t{task=}\n\t{track_head_argname=}\n\t{num_fewshot=}')
                print(f'=> Submit of {pretrained_model_name_or_path}, {task}, {track_head_argname}, {num_fewshot=}')
                command(['sbatch',
                        slurm_script,
                        pretrained_model_name_or_path, # 1st arg
                        task,                          # 2nd arg
                        results_dir,                   # 3rd arg
                        track_head_argname,               # 4th arg: track_head_type
                        str(num_fewshot)])             # 5th arg
                count += 1
                total_count += 1
                if not debug: time.sleep(2) # To avoid overloading the scheduler

                if count >= max_jobs:
                    print(f'Reached {max_jobs} limit, stopping here for now. Waiting 3 hours.\nTotal jobs submitted so far: {total_count}')
                    time.sleep(3 * 3600) # Sleep for 3 hours
                    count = 0
                    max_jobs = 100 # After the first batch, reduce the max jobs to avoid overloading the scheduler

print(f'Total jobs submitted: {total_count}')