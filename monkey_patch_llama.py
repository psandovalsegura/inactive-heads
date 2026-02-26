import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, Cache, FlashAttentionKwargs, Unpack, apply_rotary_pos_emb, repeat_kv
import transformers.models.llama.modeling_llama as modeling_llama
from monkey_patch_head_types import *

# From: transformers/models/llama/modeling_llama.py

def my_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    track_head_type: Optional[HeadType],   # @psando: if specified, we track the head type for each attention head
    zero_track_head_type: bool = False,    # @psando: if True, we zero out the output of the above specified head type
    record_head_scores: bool = False,      # @psando: if True, we record the attention scores for each head
    layers_to_exclude: list = [],          # @psando
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if track_head_type is not None and not isinstance(track_head_type, (FullHeadOutput, FullHeadOutputNormalized)) and not module.fake_batch: # @psando: create a dormant_mask boolean tensor of shape (B, N_head)
        assert not torch.isnan(attn_weights).any(), "attn_weights contains nan values and should not"
        # print(f"{attn_weights.shape=}")   # (B, N_head, S, S)
        # print(f"{value_states.shape=}")   # (B, N_head, S, D)
        # print(f"{attention_mask.shape=}") # (B, 1, 1, S)
        # print(f"{module.pad_idxs=}")      # (B,)

        if isinstance(track_head_type, RandomHeads): 
            # where an entry is True with probability zero_dormant_randomly_prob
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            selection_probability = torch.rand_like(attn_weights[:,:,0,0])
            dormant_mask = selection_probability < track_head_type.threshold
            if record_head_scores: module.head_scores.append(selection_probability.cpu())
        elif isinstance(track_head_type, (DormantHeads, NormalizedDormantHeads)):
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            attn_weights_copy = attn_weights.detach().clone() # (B, N_head, S, S)

            # Use nan causal mask to exclude zeros from average
            causal_mask = torch.tril(torch.ones_like(attn_weights_copy)).bool()
            attn_weights_copy[~causal_mask] = torch.nan
            
            # Truncate attn_weights to exclude padding (zeros) tokens
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                B, _, S, _ = attn_weights_copy.shape
                pad_idxs_tensor = torch.tensor(module.pad_idxs, device=attn_weights_copy.device).view(B, 1, 1, 1)
                indices = torch.arange(S, device=attn_weights_copy.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor                  # (B, 1, 1, S) which will broadcast
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                padding_mask = attention_mask == torch.finfo(attention_mask.dtype).min
            attn_weights_copy.masked_fill_(padding_mask, torch.nan)

            avg_weight = attn_weights_copy.nanmean(dim=-2) # (B, N_head, S)
            first_token_avg_weight = avg_weight[:,:,0] # (B, N_head)
            if isinstance(track_head_type, DormantHeads):
                dormant_mask = first_token_avg_weight > track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(first_token_avg_weight.cpu())
            elif isinstance(track_head_type, NormalizedDormantHeads):
                layer_context = first_token_avg_weight.mean(dim=1) # (B,)
                relative_first_token_avg_weight = (first_token_avg_weight / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_first_token_avg_weight > track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_first_token_avg_weight.cpu())
            else:
                ValueError(f"Unsupported track_head_type: {track_head_type}")
        elif isinstance(track_head_type, (HonorHeads, UnnormalizedHonorHeads)): 
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            norm_per_token = attn_output.norm(dim=-1) # (B, N_head, S)
            
            # padding token norms do not matter, so we set them to nan to be ignored by nanmean
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                B, _, S = norm_per_token.shape
                pad_idxs_tensor = torch.tensor(module.pad_idxs, device=norm_per_token.device).view(B, 1, 1)
                indices = torch.arange(S, device=norm_per_token.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor               # (B, 1, S) which will broadcast
                norm_per_token.masked_fill_(padding_mask, torch.nan)
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                for b in range(norm_per_token.shape[0]):
                    pad_idx = torch.sum(~(attention_mask[b,0,-1,:] == torch.finfo(attention_mask.dtype).min), dim=-1).item()
                    norm_per_token[b, :, pad_idx:] = torch.nan
                
            avg_norm_per_head = norm_per_token.nanmean(dim=-1) # (B, N_head)
            assert not torch.isnan(avg_norm_per_head).any(), "avg_norm_per_head contains nan values and should not"

            if isinstance(track_head_type, HonorHeads):
                # compute average across all heads in layer
                layer_context = avg_norm_per_head.mean(dim=1) # (B,)
                # head output defn: avg_norm_per_head < threshold_avg_weight * layer_context
                relative_avg_norm_per_head = (avg_norm_per_head / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_avg_norm_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_avg_norm_per_head.cpu())
            elif isinstance(track_head_type, UnnormalizedHonorHeads):
                dormant_mask = avg_norm_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(avg_norm_per_head.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {track_head_type}")
        elif isinstance(track_head_type, (EntropyHeads, NormalizedEntropyHeads)):
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            attn_weights_copy = attn_weights.detach().clone() # (B, N_head, S, S)
            
            # Use nan causal mask to exclude zeros from average
            causal_mask = torch.tril(torch.ones_like(attn_weights_copy)).bool()
            attn_weights_copy[~causal_mask] = torch.nan
            
            # Truncate attn_weights to exclude padding (zeros) tokens
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                B, _, S, _ = attn_weights_copy.shape
                pad_idxs_tensor = torch.tensor(module.pad_idxs, device=attn_weights_copy.device).view(B, 1, 1, 1)
                indices = torch.arange(S, device=attn_weights_copy.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor                  # (B, 1, 1, S) which will broadcast
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                padding_mask = attention_mask == torch.finfo(attention_mask.dtype).min
            attn_weights_copy.masked_fill_(padding_mask, torch.nan)

            entropy_per_head = -(attn_weights_copy * torch.log(attn_weights_copy)).nansum(dim=-1) # (B, N_head, S)
            avg_entropy_per_head = entropy_per_head.nanmean(dim=-1) # (B, N_head)
            assert not torch.isnan(avg_entropy_per_head).any(), "avg_entropy_per_head contains nan values and should not"

            if isinstance(track_head_type, EntropyHeads):
                dormant_mask = avg_entropy_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(avg_entropy_per_head.cpu())
            elif isinstance(track_head_type, NormalizedEntropyHeads):
                layer_context = avg_entropy_per_head.mean(dim=1) # (B,)
                relative_avg_entropy_per_head = (avg_entropy_per_head / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_avg_entropy_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_avg_entropy_per_head.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {track_head_type}")
        elif isinstance(track_head_type, (ValueVectorMagnitudeFirstToken, ValueVectorMagnitudeNormalizedFirstToken)):
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            value_norm_per_token = value_states.norm(dim=-1) # (B, N_head, S)

            # padding token norms do not matter, but because we don't average over tokens, we don't need to set them to nan
            first_token_value_norm = value_norm_per_token[:,:,0] # (B, N_head)
            assert not torch.isnan(first_token_value_norm).any(), "first_token_value_norm contains nan values and should not"

            if isinstance(track_head_type, ValueVectorMagnitudeFirstToken):
                dormant_mask = first_token_value_norm < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(first_token_value_norm.cpu())
            elif isinstance(track_head_type, ValueVectorMagnitudeNormalizedFirstToken):
                layer_context = first_token_value_norm.mean(dim=1) # (B,)
                relative_first_token_value_norm = (first_token_value_norm / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_first_token_value_norm < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_first_token_value_norm.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {track_head_type}")
        elif isinstance(track_head_type, (ValueVectorAvgMagnitude, ValueVectorAvgNormalizedMagnitude)):
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            value_norm_per_token = value_states.norm(dim=-1) # (B, N_head, S)
            
            # padding token norms do not matter, so we set them to nan to be ignored by nanmean
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                B, _, S = value_norm_per_token.shape
                pad_idxs_tensor = torch.tensor(module.pad_idxs, device=value_norm_per_token.device).view(B, 1, 1)
                indices = torch.arange(S, device=value_norm_per_token.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor               # (B, 1, S) which will broadcast
                value_norm_per_token.masked_fill_(padding_mask, torch.nan)
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                for b in range(value_norm_per_token.shape[0]):
                    pad_idx = torch.sum(~(attention_mask[b,0,-1,:] == torch.finfo(attention_mask.dtype).min), dim=-1).item()
                    value_norm_per_token[b, :, pad_idx:] = torch.nan

            avg_value_norm_per_head = value_norm_per_token.nanmean(dim=-1) # (B, N_head)
            if isinstance(track_head_type, ValueVectorAvgMagnitude):
                dormant_mask = avg_value_norm_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(avg_value_norm_per_head.cpu())
            elif isinstance(track_head_type, ValueVectorAvgNormalizedMagnitude):
                layer_context = avg_value_norm_per_head.mean(dim=1) # (B,)
                relative_avg_value_norm_per_head = (avg_value_norm_per_head / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_avg_value_norm_per_head < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_avg_value_norm_per_head.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {track_head_type}")
        elif isinstance(track_head_type, (HeadOutputMagnitudeLastToken, HeadOutputMagnitudeNormalizedLastToken, HeadOutputMagnitudeNormalizedHeadLastToken)):
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            norm_per_token = attn_output.norm(dim=-1) # (B, N_head, S)
            B, _, S = norm_per_token.shape
            
            # padding token norms do not matter, so we set them to nan to be ignored by nanmean
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                pad_idxs_tensor = torch.tensor(module.pad_idxs, device=norm_per_token.device) # (B,)
                indices = torch.arange(S, device=norm_per_token.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor.view(B, 1, 1) # (B, 1, S) which will broadcast
                norm_per_token.masked_fill_(padding_mask, torch.nan)
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                pad_idxs_tensor = torch.zeros(B, dtype=torch.long, device=norm_per_token.device) 
                for b in range(B):
                    pad_idx = torch.sum(~(attention_mask[b,0,-1,:] == torch.finfo(attention_mask.dtype).min), dim=-1).item()
                    pad_idxs_tensor[b] = pad_idx
                    norm_per_token[b, :, pad_idx:] = torch.nan
        
            # get the last (non-padding) token norm
            last_token_norm = norm_per_token[torch.arange(B), :, pad_idxs_tensor - 1] # (B, N_head)

            if isinstance(track_head_type, HeadOutputMagnitudeLastToken):
                dormant_mask = last_token_norm < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(last_token_norm.cpu())
            elif isinstance(track_head_type, HeadOutputMagnitudeNormalizedHeadLastToken):
                layer_context = norm_per_token.nanmean(dim=-1) # (B, N_head)
                relative_last_token_norm = (last_token_norm / layer_context) # (B, N_head) elementwise division
                dormant_mask = relative_last_token_norm < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_last_token_norm.cpu()) 
            elif isinstance(track_head_type, HeadOutputMagnitudeNormalizedLastToken):
                layer_context = norm_per_token.nanmean(dim=-1).mean(dim=1) # (B,)
                relative_last_token_norm = (last_token_norm / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_last_token_norm < track_head_type.threshold # (B, N_head)
                if record_head_scores: module.head_scores.append(relative_last_token_norm.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {track_head_type}")
        else:
            raise ValueError(f"Unsupported track_head_type: {track_head_type}")

        if dormant_mask.any():
            module.dormant_masks.append(dormant_mask.cpu())
            # Heads that are dropped will not change the original hidden state at those head dimensions
            # so we only execute attn_weights @ value_states for heads that are not dropped
            # i.e. attn_output will be the same as hidden_states for dropped heads
            #                  will be the result of attn_weights @ value_states for non-dropped heads
            # 3. Compute the attention output 
            # start with zeros, replace with hidden_states for dropped heads
            if zero_track_head_type:
                attn_output[dormant_mask] = 0
            # else: attn_output has already been computed above so we use it without modification
        else:
            module.dormant_masks.append(torch.zeros((attn_weights.shape[0], attn_weights.shape[1]), dtype=bool))
            attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class MyLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int,
                 save_value_states=False, 
                 save_pre_output_proj_hidden_states=False, 
                 save_post_output_proj_hidden_states=False,
                 track_head_type=None,
                 zero_track_head_type=False,
                 record_head_scores=False,
                 layers_to_exclude=[]):
        super().__init__(config=config, layer_idx=layer_idx)

        # @psando
        self.save_value_states = save_value_states
        self.save_pre_output_proj_hidden_states = save_pre_output_proj_hidden_states
        self.save_post_output_proj_hidden_states = save_post_output_proj_hidden_states
        self.value_states = None
        self.pre_output_proj_hidden_states = None
        self.post_output_proj_hidden_states = None

        self.track_head_type = track_head_type
        self.zero_track_head_type = zero_track_head_type
        self.record_head_scores = record_head_scores
        self.layers_to_exclude = layers_to_exclude

        self.dormant_masks = []
        self.head_scores = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.save_value_states:           # @psando
            self.value_states = value_states # @psando

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = my_eager_attention_forward # @psando

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            track_head_type=self.track_head_type,
            zero_track_head_type=self.zero_track_head_type,
            record_head_scores=self.record_head_scores,
            layers_to_exclude=self.layers_to_exclude,
            **kwargs,
        )
        if self.save_pre_output_proj_hidden_states:                         # @psando
            self.pre_output_proj_hidden_states = attn_output.transpose(1,2) # @psando
        if isinstance(self.track_head_type, (FullHeadOutput, FullHeadOutputNormalized)):
            # attn_output has shape (B, S, N_head, head_dim)
            # o_proj weight will have shape (N_head, head_dim, D_model)
            hidden_size = self.o_proj.weight.shape[0]
            num_attention_heads, head_dim = attn_output.shape[2:]
            o_proj_weight = self.o_proj.weight.view(hidden_size, num_attention_heads, head_dim).permute(1,2,0)

            # compute head outputs (using slice of o_proj weight for each head)
            head_outputs = torch.einsum('bsnh,nhd->bsnd', attn_output, o_proj_weight) # (B, S, N_head, D_model)
            norm_per_token = head_outputs.norm(dim=-1) # (B, S, N_head)
            norm_per_token = norm_per_token.transpose(1,2) # (B, N_head, S)

            # padding token norms do not matter, so we set them to nan to be ignored by nanmean
            if hasattr(self, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                B, _, S = norm_per_token.shape
                pad_idxs_tensor = torch.tensor(self.pad_idxs, device=norm_per_token.device).view(B, 1, 1)
                indices = torch.arange(S, device=norm_per_token.device) # (S,)
                padding_mask = indices >= pad_idxs_tensor               # (B, 1, S) which will broadcast
                norm_per_token.masked_fill_(padding_mask, torch.nan)
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                for b in range(norm_per_token.shape[0]):
                    pad_idx = torch.sum(~(attention_mask[b,0,-1,:] == torch.finfo(attention_mask.dtype).min), dim=-1).item()
                    norm_per_token[b, :, pad_idx:] = torch.nan
                
            avg_norm_per_head = norm_per_token.nanmean(dim=-1) # (B, N_head)
            assert not torch.isnan(avg_norm_per_head).any(), "avg_norm_per_head contains nan values and should not"

            # verify manual_output and standard_output are the same
            # manual_output = head_outputs.sum(dim=2)
            # standard_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
            # print(f"manual_output equals standard_output: {torch.allclose(manual_output, standard_output, atol=1e-5)}")

            if isinstance(self.track_head_type, FullHeadOutput):
                dormant_mask = avg_norm_per_head < self.track_head_type.threshold # (B, N_head)
                if self.record_head_scores: self.head_scores.append(avg_norm_per_head.cpu())
            elif isinstance(self.track_head_type, FullHeadOutputNormalized):
                layer_context = avg_norm_per_head.mean(dim=1) # (B,)
                relative_avg_norm_per_head = (avg_norm_per_head / layer_context[:, None]) # (B, N_head)
                dormant_mask = relative_avg_norm_per_head < self.track_head_type.threshold # (B, N_head)
                if self.record_head_scores: self.head_scores.append(relative_avg_norm_per_head.cpu())
            else:
                raise ValueError(f"Unsupported track_head_type: {self.track_head_type}")
            
            if dormant_mask.any():
                self.dormant_masks.append(dormant_mask.cpu())
                # Permute attn_output from (B, S, N_head, head_dim) to (B, N_head, S, head_dim) for easier indexing
                attn_output = attn_output.transpose(1,2) # (B, N_head, S, head_dim)
                attn_output[dormant_mask] = 0
                # Permute back to (B, S, N_head, head_dim)
                attn_output = attn_output.transpose(1,2) # (B, S, N_head, head_dim)
            else:
                self.dormant_masks.append(torch.zeros((attn_weights.shape[0], attn_weights.shape[1]), dtype=bool))

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.save_post_output_proj_hidden_states:          # @psando
            self.post_output_proj_hidden_states = attn_output # @psando
        return attn_output, attn_weights
    
def patch_llama(save_value_states=False, 
                save_pre_output_proj_hidden_states=False, 
                save_post_output_proj_hidden_states=False,
                track_head_type=None,
                zero_track_head_type=False,
                record_head_scores=False,
                layers_to_exclude=[]):
    class PatchedLlamaAttention(MyLlamaAttention):
        def __init__(self, config: LlamaConfig, layer_idx: int, *args, **kwargs):
            assert config._attn_implementation == 'eager', "Only eager attention is supported because access to intermediate tensors like attention weights is needed.\n" \
                   "Please set `AutoModelForCausalLM.from_pretrained(..., attn_implementation='eager')` when initializing the model."
            super().__init__(
                config,
                layer_idx=layer_idx,
                save_value_states=save_value_states,
                save_pre_output_proj_hidden_states=save_pre_output_proj_hidden_states,
                save_post_output_proj_hidden_states=save_post_output_proj_hidden_states,
                track_head_type=track_head_type,
                zero_track_head_type=zero_track_head_type,
                record_head_scores=record_head_scores,
                layers_to_exclude=layers_to_exclude,
                *args,
                **kwargs
            )

    modeling_llama.LlamaAttention = PatchedLlamaAttention
