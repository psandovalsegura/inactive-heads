import torch
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

def collect_saved_tensors(model):
    """
    Collects saved tensors from a model.
    Args:
        model: Model
        get_layers_fn: Callable[[ModelClass], List[LayerClass]] given a model, returns the list of layers
        get_attn_fn: Callable[[LayerClass], AttentionClass] given a layer, returns the attention module
    """
    get_layers_fn, get_attn_fn = get_model_accessors(model)
    
    # Collect tensors from attention module
    value_states = []
    pre_output_proj_hidden_states = []
    post_output_proj_hidden_states = []
    for layer in get_layers_fn(model):
        attention_module = get_attn_fn(layer)
        if attention_module.save_value_states:
            value_states.append(attention_module.value_states.detach().cpu())
            del attention_module.value_states
        if attention_module.save_pre_output_proj_hidden_states:
            pre_output_proj_hidden_states.append(attention_module.pre_output_proj_hidden_states.detach().cpu())
            del attention_module.pre_output_proj_hidden_states
        if attention_module.save_post_output_proj_hidden_states:
            post_output_proj_hidden_states.append(attention_module.post_output_proj_hidden_states.detach().cpu())
            del attention_module.post_output_proj_hidden_states
        
    torch.cuda.empty_cache()

    return SavedTensors(model_sizes=get_model_sizes(model),
                        value_states=(value_states if value_states else None), 
                        pre_output_proj_hidden_states=(pre_output_proj_hidden_states if pre_output_proj_hidden_states else None),
                        post_output_proj_hidden_states=(post_output_proj_hidden_states if post_output_proj_hidden_states else None))
class SavedTensors:
    def __init__(self, 
                 model_sizes,
                 value_states=None, 
                 pre_output_proj_hidden_states=None, 
                 post_output_proj_hidden_states=None):
        """
        Checks that saved tensors have expected sizes before creating object. Specifically, checks that:
            - value_states: List[Tensor[batch_size, num_key_value_heads, seq_len, head_dim]]
            - pre_output_proj_hidden_states: List[Tensor[batch_size, num_heads, seq_len, head_dim]]
            - post_output_proj_hidden_states: List[Tensor[batch_size, seq_len, hidden_size]]
        Args:
            model_sizes: Tuple[int, int, int, int, int]
        """
        num_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_size = model_sizes
        if value_states:
            assert len(value_states) == num_layers, f"{len(value_states)=} layers, expected {num_layers}"
        if pre_output_proj_hidden_states:
            assert len(pre_output_proj_hidden_states) == num_layers, f"{len(pre_output_proj_hidden_states)=} layers, expected {num_layers}"
        if post_output_proj_hidden_states:
            assert len(post_output_proj_hidden_states) == num_layers, f"{len(post_output_proj_hidden_states)=} layers, expected {num_layers}"
        for i in range(num_layers):
            if value_states:
                assert value_states[i].shape[1] == num_key_value_heads, f"{value_states[i].shape=}, expected {num_key_value_heads}"
                assert value_states[i].shape[3] == head_dim, f"{value_states[i].shape=}, expected {head_dim}"
            if pre_output_proj_hidden_states:
                assert pre_output_proj_hidden_states[i].shape[1] == num_attention_heads, f"{pre_output_proj_hidden_states[i].shape=}, expected {num_attention_heads}"
                assert pre_output_proj_hidden_states[i].shape[3] == head_dim, f"{pre_output_proj_hidden_states[i].shape=}, expected {head_dim}"
            if post_output_proj_hidden_states:
                assert post_output_proj_hidden_states[i].shape[2] == hidden_size, f"{post_output_proj_hidden_states[i].shape=}, expected {hidden_size}"
        self.value_states = value_states
        self.pre_output_proj_hidden_states = pre_output_proj_hidden_states
        self.post_output_proj_hidden_states = post_output_proj_hidden_states

def get_inputs_embeds(model, input_ids):
    if any(isinstance(model, model_class) for model_class in [Olmo2ForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM]):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids)
    else:
        raise ValueError(f"Unsupported model type: {model.__class__}")
    
def get_model_sizes(model):
    """
    Returns a tuple with sizes of the model.
    Args:
        model: Model
    Returns:
        returns (num_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_size)
    """
    if any(isinstance(model, model_class) for model_class in [Olmo2ForCausalLM, Qwen2ForCausalLM]):
        assert all([model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, model.config.hidden_size])
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        return (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, head_dim, model.config.hidden_size)
    elif isinstance(model, GPT2LMHeadModel):
        assert all([model.config.n_layer, model.config.n_head, model.config.n_embd])
        head_dim = model.config.n_embd // model.config.n_head
        return (model.config.n_layer, model.config.n_head, model.config.n_head, head_dim, model.config.n_embd)
    elif isinstance(model, LlamaForCausalLM):
        assert all([model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim, model.config.hidden_size])
        return (model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim, model.config.hidden_size)
    else:
        raise ValueError(f"Unsupported model type: {model.__class__}")

def get_model_accessors(model):
    if any(isinstance(model, model_class) for model_class in [Olmo2ForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM]):
        get_layers_fn = lambda model: model.model.layers
        get_attn_fn = lambda layer: layer.self_attn
    elif isinstance(model, GPT2LMHeadModel):
        get_layers_fn = lambda model: model.transformer.h
        get_attn_fn = lambda layer: layer.attn
    else:
        raise ValueError(f"Unsupported model type: {model.__class__}")
    return get_layers_fn, get_attn_fn

def get_dormant_proportion(model, forward_pass_idx=None, return_dormant_counts=False, verbose=False):
    """
    Compute the dormant proportion averaged over a number of forward passes.
    Or if forward_pass_idx is provided, compute the dormant proportion for that specific forward pass.
    """
    get_layers_fn, get_attn_fn = get_model_accessors(model)
    num_layers, num_attention_heads, _, _, _ = get_model_sizes(model)
    # Get the dormant masks for every layer
    # self_attn.dormant_masks is a list of len num forward passes, each element of shape (B, N_head)
    if forward_pass_idx is not None:
        for layer in get_layers_fn(model):
            dormant_mask_shape = get_attn_fn(layer).dormant_masks[forward_pass_idx].shape
            assert dormant_mask_shape[0] == 1, f"forward_pass_idx must be for a single sample, but got {dormant_mask_shape=}"
        num_samples = dormant_mask_shape[0]
        # index into list to get (1, N_head) for each layer
        counts_per_layer = [get_attn_fn(layer).dormant_masks[forward_pass_idx] for layer in get_layers_fn(model)]
    else:
        num_samples = torch.cat(get_attn_fn(get_layers_fn(model)[0]).dormant_masks, dim=0).shape[0]
        # concatenate along first dimension to get (num_samples, N_head), sum along first dimension to get counts for every head of this layer (1, N_head)
        counts_per_layer = [torch.cat(get_attn_fn(layer).dormant_masks, dim=0).sum(dim=0, keepdim=True) for layer in get_layers_fn(model)]
    # print(f"{len(counts_per_layer)=}") # N_layers 
    # print(f"{counts_per_layer[0].shape=}") # (1, N_head)

    # concatenate counts of each layer to get (num_layers, num_attention_heads)
    full_model_counts = torch.concat(counts_per_layer, axis=0) # (num_layers, num_attention_heads)
    if verbose:
        print(f"Avg dormant proportion over {num_samples=}")
    avg_proportion_dormant = full_model_counts.sum() / (num_layers * num_attention_heads * num_samples)
    if return_dormant_counts:
        return avg_proportion_dormant.item(), full_model_counts
    else:
        return avg_proportion_dormant.item()
    
def get_head_scores(model, forward_pass_idx=None):
    get_layers_fn, get_attn_fn = get_model_accessors(model)
    # Get the head scores for every layer
    # self_attn.head_scores is a list of len num forward passes, each element of shape (B, N_head)
    if forward_pass_idx is not None:
        for layer in get_layers_fn(model):
            self_attn = get_attn_fn(layer)
            assert self_attn.record_head_scores, f"Head scores not recorded in self-attention module. Use patch_model(record_head_scores=True,...)"
            head_scores_shape = self_attn.head_scores[forward_pass_idx].shape
            assert head_scores_shape[0] == 1, f"forward_pass_idx must be for a single sample, but got {head_scores_shape=}"
        # index into list to get (1, N_head) for each layer
        scores_per_layer = [get_attn_fn(layer).head_scores[forward_pass_idx] for layer in get_layers_fn(model)]
    else:
        # concatenate along first dimension to get (num_samples, N_head), for every layer
        scores_per_layer = [torch.cat(get_attn_fn(layer).head_scores, dim=0) for layer in get_layers_fn(model)]
    # Return a (num_layers * num_samples, N_head) set of scores
    return torch.cat(scores_per_layer, dim=0)

from monkey_patch_llama import patch_llama
from monkey_patch_olmo2 import patch_olmo2
from monkey_patch_qwen2 import patch_qwen2    
from monkey_patch_gpt2 import patch_gpt2
from monkey_patch_head_types import is_supported_head_type

def should_apply_chat_template(pretrained_model_name_or_path):
    return any(chat_str in pretrained_model_name_or_path for chat_str in ['-Instruct', '-chat', '-SFT', '-DPO'])

def patch_model(pretrained_model_name_or_path, 
                save_value_states=False, 
                save_pre_output_proj_hidden_states=False, 
                save_post_output_proj_hidden_states=False,
                track_head_type=None,       # if None, heads are not tracked (i.e. we do not construct a layer's head masks)
                zero_track_head_type=False, # if True, tracked heads will be zeroed
                record_head_scores=False,   # if True, tracked heads scores will be saved
                layers_to_exclude=[]):
    assert not (zero_track_head_type and (save_value_states or save_pre_output_proj_hidden_states or save_post_output_proj_hidden_states)), \
           "Cannot save tensors and zero tracked heads at the same time because pre_output_proj_hidden_states tensors " \
           "will be zeroed and subsequent tensors will not be true to the original forward pass."
    if track_head_type or record_head_scores:
        assert is_supported_head_type(track_head_type), f"Unsupported head type: {track_head_type}. See is_supported_head_type in monkey_patch_head_types.py"
    if zero_track_head_type:
        assert track_head_type, "Cannot zero tracked heads if not tracking any heads. Set track_head_type."

    if any(llama_str in pretrained_model_name_or_path for llama_str in ['Llama-2', 'Llama-3.1', 'Llama-3.2']):
        patch_fn = patch_llama
    elif any(olmo_str in pretrained_model_name_or_path for olmo_str in ['OLMo-2']):
        patch_fn = patch_olmo2
    elif any(qwen_str in pretrained_model_name_or_path for qwen_str in ['Qwen2.5']):
        patch_fn = patch_qwen2
    elif pretrained_model_name_or_path in ['gpt2']:
        patch_fn = patch_gpt2
    else:
        raise NotImplementedError(f"Patching not implemented for {pretrained_model_name_or_path}")
    patch_fn(save_value_states=save_value_states,
             save_pre_output_proj_hidden_states=save_pre_output_proj_hidden_states,
             save_post_output_proj_hidden_states=save_post_output_proj_hidden_states,
             track_head_type=track_head_type,
             zero_track_head_type=zero_track_head_type,
             record_head_scores=record_head_scores,
             layers_to_exclude=layers_to_exclude)
    
