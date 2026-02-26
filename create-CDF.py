# %%
import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# set default type to float16
# torch.set_default_dtype(torch.float16) # TODO: comment out
# set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from monkey_patch_head_types import HeadType, x_intercept_for_cdf
model_ids = ['Qwen/Qwen2.5-7B', 'meta-llama/Llama-3.1-8B', 'allenai/OLMo-2-1124-7B', 'meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',\
            'allenai/OLMo-2-1124-7B-SFT', 'allenai/OLMo-2-1124-7B-DPO', 'allenai/OLMo-2-1124-7B-Instruct',\
            'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B',  'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B']
head_types = ['DormantHeads:0.0', 'HonorHeads:0.0', 'NormalizedDormantHeads:0.0', 'UnnormalizedHonorHeads:0.0', \
              'EntropyHeads:0.0', 'NormalizedEntropyHeads:0.0']
head_types += ['ValueVectorMagnitudeFirstToken:0.0', 'ValueVectorMagnitudeNormalizedFirstToken:0.0',\
               'ValueVectorAvgMagnitude:0.0', 'ValueVectorAvgNormalizedMagnitude:0.0', \
               'HeadOutputMagnitudeLastToken:0.0', 'HeadOutputMagnitudeNormalizedLastToken:0.0', 'HeadOutputMagnitudeNormalizedHeadLastToken:0.0', \
               'FullHeadOutput:0.0', 'FullHeadOutputNormalized:0.0']
head_types = [HeadType.from_string(s) for s in head_types]
print(f'{len(model_ids)=}')
print(f'{len(head_types)=}')

# %%
results_dir = Path('results-record-head-scores')
new_cdf_dir = Path('results-head-score-cdfs')
save_new_cdfs = True
task_id = 'mmlu'

# Create new CDFs
for model_id in tqdm(model_ids, desc="Models"): # [11:12] for Qwen2.5-7B only
    for head_type in head_types:
        model_name = model_id.replace('/', '_')
        head_scores_path = results_dir / model_name / task_id / 'head_scores' / f'{head_type.argname}.pt'
        new_cdf_path = new_cdf_dir / task_id / f'{model_name}_{head_type.name}.pt'
        if head_scores_path.exists() and not new_cdf_path.exists(): # only compute/save if it doesn't already exist
            print(f'{model_name=} {head_type.name=}')
            # Create new CDF
            head_scores = torch.load(head_scores_path, weights_only=False).flatten().float().cpu()
            print(f'\t{head_scores.shape=}')
            sorted_scores = torch.sort(head_scores).values
            new_cdf = torch.stack((sorted_scores, torch.arange(1, len(sorted_scores) + 1) / len(sorted_scores)), dim=1)
            # Ensure there are no infs or nans
            assert not torch.any(torch.isinf(new_cdf)), f"CDF contains inf values. Cannot create from {head_scores_path}."
            assert not torch.any(torch.isnan(new_cdf)), f"CDF contains NaN values. Cannot create from {head_scores_path}."

            if save_new_cdfs: 
                new_cdf_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(new_cdf, new_cdf_path)
                print(f'\tSaved new CDF to {new_cdf_path}')
            # print(f'{new_cdf.shape=}, {new_cdf.dtype=}')
