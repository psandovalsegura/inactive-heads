import os
import json
import torch
import lm_eval
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_memlab.utils import readable_size
from lm_eval.models.utils import stop_sequences_criteria

from monkey_patch_utils import patch_model, get_model_sizes, get_model_accessors, should_apply_chat_template, get_head_scores
from monkey_patch_head_types import HeadType

def find_padding_start(t: torch.Tensor) -> int:
    """
    Returns the index of the first 0 in t.
    lm_eval.models.utils.pad_and_concat uses 0 to pad during HFLM._loglikelihood_tokens
    which is called when evaluating using loglikelihood of continuations for MC questions.
    """
    # Ensure the tensor is 1D
    assert t.dim() == 1, "Tensor must be 1D"
    
    # Find indices where elements are non-zero
    nonzero_indices = torch.nonzero(t != 0, as_tuple=False)
    assert nonzero_indices.numel() > 0, "Tensor must contain at least one non-zero element"

    # Get the maximum index of non-zero elements
    j = nonzero_indices[:, 0].max().item()

    # Padding starts at the next index
    pad_idx = j + 1
    assert torch.all(t[pad_idx:] == 0), "all tokens after pad_idx should be 0"
    return pad_idx

def check_fake_batch(t: torch.Tensor) -> bool:
    """
    Returns True if the tensor is a dummy batch of size containing all 1 values
    from lm_eval HFLM._detect_batch_size.
    """
    return torch.all(t == 1)

class PaddingSaverHFLM(lm_eval.models.huggingface.HFLM):
    """
    HFLM that saves padding indices to all self-attention layers, so that we can properly 
    truncate each attention matrix. This is necessary because HFLM._loglikelihood_tokens
    pads the input tokens with zero but does not construct appropriate attention masks.
    """
    def __init__(self, *args, get_layers_fn=None, get_attn_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_counts = [] # not using name batch_sizes to avoid conflict with lm_eval.models.huggingface.HFLM.batch_sizes
        assert get_layers_fn is not None and get_attn_fn is not None, "get_layers_fn and get_attn_fn must be provided"
        self.get_layers_fn = get_layers_fn
        self.get_attn_fn = get_attn_fn

    def _set_fake_batch(self, flag):
        """
        Save bool that current batch is fake (See HFLM._detect_batch_size) to prevent attention layers from saving statistics
        """
        for layer in self.get_layers_fn(self.model):
            self_attn = self.get_attn_fn(layer)
            self_attn.fake_batch = flag

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        Args:
            inps: torch.Tensor of shape (B, S)
        """
        is_fake_batch = check_fake_batch(inps)
        if is_fake_batch:
            self._set_fake_batch(True)
        else:
            # debug HFLM.loglikelihood_rolling:
            # print(f'Input shape: {inps.shape}')
            # print(self.tokenizer.decode(inps[0]))
            # import pdb; pdb.set_trace()
            # print(f'[_model_call] number of forward passes so far: {len(self.model.model.layers[0].self_attn.dormant_masks)}')
            # import pdb; pdb.set_trace()
            self.batch_counts.append(inps.size(0))
            assert attn_mask is None, "attn_mask should be None when calling HFLM._loglikelihood_tokens"
            pad_idxs = torch.tensor([find_padding_start(inp) for inp in inps]) # (B,)
            # save pad_idxs to all self-attention layers
            for layer in self.get_layers_fn(self.model):
                self_attn = self.get_attn_fn(layer)
                self_attn.pad_idxs = pad_idxs
                self_attn.fake_batch = False

        # _model_call is used by HFLM._detect_batch_size and if an OOM occurs we would
        # not be able to _set_fake_batch(False) after the forward, so we use a try-except block
        # to catch the exception, _set_fake_batch(False), then re-raise the exception
        try:
            out = super()._model_call(inps=inps, attn_mask=attn_mask, labels=labels)
            # after this forward pass, we set fake_batch to False so that following calls are not affected
            if is_fake_batch:
                self._set_fake_batch(False)
            return out
        except RuntimeError as e:
            # after this forward pass, we set fake_batch to False so that following calls are not affected
            if is_fake_batch:
                self._set_fake_batch(False)
            raise
    
    # From lm_eval/models/huggingface.py
    # One change: Disable cache so that all attention weights can be accessed on each forward
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        # @psando: explicit check on max_length kwarg passed to _model_generate() in HFLM.generate_until
        assert self.max_gen_toks == 256, "To ensure fair evaluation, all models should have max_gen_toks set to 256 in HFLM" 
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False, # @psando: changed to False
            **generation_kwargs,
        )


def main():
    def none_or_float(value):
        if value.lower() == 'none':
            return None
        return float(value)
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('tasks', type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=str, default='auto')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--save_generated_samples', action='store_true', help='Save generated samples in results json')
    parser.add_argument('--save_recorded_head_scores', action='store_true', help='Save recorded head scores as .pt files in results dir')
    parser.add_argument('--track_head_type', type=HeadType.from_string, default=None, 
                        help="Specify the head type to track and its parameter, e.g., 'DormantHeads:0.5' or 'None'")
    parser.add_argument('--dont_zero_dormant', action='store_true', help='Do not zero dormant heads')
    def parse_layers(s):
        if s.strip() == "":
            return []
        return [int(item.strip()) for item in s.split(',')]
    parser.add_argument('--layers_to_exclude', type=parse_layers, default=[], help='Layers to exclude from being dropped (0-indexed and separated by commas)')
    parser.add_argument('--get_baseline', action='store_true', help='Get baseline results without zeroing heads')
    args = parser.parse_args()
    print(args)
    assert args.device == 'cuda', "only cuda is supported"
    assert len(args.tasks) == 1, "only 1 task is supported, otherwise proportions will be over all task inputs"
    if args.save_recorded_head_scores:
        assert args.track_head_type is not None, "When --save_recorded_head_scores is set, --track_head_type must be specified"
        assert args.dont_zero_dormant is True, "When --save_recorded_head_scores is set, --dont_zero_dormant must be True"

    if not args.get_baseline:
        patch_model(args.pretrained_model_name_or_path,
                    track_head_type=args.track_head_type,
                    zero_track_head_type=(not args.dont_zero_dormant),
                    record_head_scores=args.save_recorded_head_scores,
                    layers_to_exclude=args.layers_to_exclude)
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    if any(olmo_str in args.pretrained_model_name_or_path for olmo_str in ['OLMo-2']):
        # OLMo loaded in float16, otherwise inference is too slow
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, attn_implementation='eager', torch_dtype=torch.float16, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, attn_implementation='eager', torch_dtype="auto")
        model.to(args.device)
    # model.half() # using half precision causes nan in Llama-2 inference at last layer input_layernorm (use only for testing on small GPUs)
    num_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_size = get_model_sizes(model)
    get_layers_fn, get_attn_fn = get_model_accessors(model)
    apply_chat_template = should_apply_chat_template(args.pretrained_model_name_or_path)
    print(f"=> Applying chat template: {apply_chat_template}")
    print(f"=> Attention implementation: {model.config._attn_implementation}")
    print(f"=> Attention class: {type(get_attn_fn(layer=(get_layers_fn(model)[0])))}")
    print(f"=> Model size: {readable_size(torch.cuda.max_memory_allocated())}")
    print(f"=> Model dtype: {model.dtype}")
    if model.generation_config.max_new_tokens:
        # if model has a generation_config with a max_new_tokens key, remove it.
        # HFLM will configure a max_length instead. Otherwise the following error will occur:
        #   Both `max_new_tokens` (=##) and `max_length`(=##) seem to have been set. `max_new_tokens` will take precedence.
        model.generation_config.max_new_tokens = None
        print(f"=> Removed max_new_tokens from generation_config")

    # set model to have use_cache=False on forward to prevent conflict with present_key_value
    # which is not present when SA is dropped
    model.config.use_cache = False
    lm_obj = PaddingSaverHFLM(pretrained=model,
                              tokenizer=tokenizer,
                              batch_size=args.batch_size,
                              get_layers_fn=get_layers_fn,
                              get_attn_fn=get_attn_fn)
        
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=apply_chat_template,
        device=args.device,
    )

    # Alternatively:
    # results = lm_eval.simple_evaluate(
    #     model='huggingface',
    #     model_args=f'pretrained={args.pretrained_model_name_or_path},attn_implementation=eager',
    #     tasks=args.tasks,
    #     num_fewshot=args.num_fewshot,
    #     batch_size=args.batch_size,
    #     device=args.device,
    # )

    # Create results directory and filename
    results_dir = Path(args.results_dir) / args.pretrained_model_name_or_path.replace('/', '_') / args.tasks[0]
    os.makedirs(results_dir, exist_ok=True)

    if args.get_baseline:
        results_file = results_dir / 'baseline.json'
    else:
        results_file = results_dir / f'{args.track_head_type}{"_dont_zero" if args.dont_zero_dormant else ""}.json'

    if args.save_recorded_head_scores and not args.get_baseline:
        # Save recorded head scores by model_id and head type
        head_scores_dir = results_dir / 'head_scores' 
        os.makedirs(head_scores_dir, exist_ok=True)
        head_scores = get_head_scores(model)
        head_scores_file = head_scores_dir / f'{args.track_head_type}.pt'
        torch.save(head_scores, head_scores_file)
        print(f"=> Saved recorded head scores to {head_scores_file}")

    # Configure results file: remove samples if not saving them and add dormant heads proportions
    if not args.save_generated_samples:
        results.pop('samples')
    if not args.get_baseline:
        # Asserts on saved dormant_masks
        assert hasattr(get_attn_fn(get_layers_fn(model)[0]), 'dormant_masks'), f"model.model.layers[0].self_attn should have dormant_masks attribute. Ensure construct_dormant=True in patch_model()"
        num_samples = torch.cat(get_attn_fn(get_layers_fn(model)[0]).dormant_masks, dim=0).shape[0]
        for layer_idx, layer in enumerate(get_layers_fn(model)):
            # All layers should have processed the same number of samples
            layer_num_samples = torch.cat(get_attn_fn(layer).dormant_masks, dim=0).shape[0]
            assert layer_num_samples == num_samples, f"model.model.layers[{layer_idx}].self_attn should have processed the same number of samples as the first layer, but got {layer_num_samples} != {num_samples}"
        # All layers should have executed the same number of forward passes (which is the length of the dormant_masks list)
        num_forward_passes = [len(get_attn_fn(layer).dormant_masks) for layer in get_layers_fn(model)]
        assert len(set(num_forward_passes)) == 1, f"all layers should have executed the same number of forward passes, but got {num_forward_passes}"

        # Calculate dormant heads proportion for each layer
        layers_to_proportion = {}
        for layer_idx, layer in enumerate(get_layers_fn(model)):
            num_dormant_heads = torch.cat(get_attn_fn(layer).dormant_masks, dim=0) # (num_samples, num_attention_heads)
            layers_to_proportion[layer_idx] = num_dormant_heads.sum().item() / (num_samples * num_attention_heads)

        # The following calculation is equivalent to np.mean(list(layers_to_proportion.values()))
        # total_dormant_heads = 0
        # for layer_idx, layer in enumerate(get_layers_fn(model)):
        #     total_dormant_heads += np.sum([dm.sum() for dm in get_attn_fn(layer).dormant_masks])
        # print(f"=> Full model dormant heads proportion (my calculation): {total_dormant_heads / (np.sum(lm_obj.batch_counts) * num_layers * num_attention_heads)}")

        # Save dormant heads proportion results to results
        results['zero_dormant'] = not args.dont_zero_dormant
        results['track_head_type'] = args.track_head_type
        results['layers_to_proportion'] = layers_to_proportion
        results['model_proportion'] = np.mean(list(layers_to_proportion.values()))
        results['layers_to_exclude'] = args.layers_to_exclude
        results['num_forward_passes'] = num_forward_passes[0]  # equal to number of forward passes
        results['num_samples'] = num_samples                   # equal to number of samples
        results['batch_size'] = args.batch_size
        
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=lm_eval.utils.handle_non_serializable, ensure_ascii=False)

if __name__ == "__main__":
    main()