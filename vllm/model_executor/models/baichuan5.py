# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only BaiChuan model compatible with HuggingFace weights."""
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, row_parallel_weight_loader, sharded_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

from .interfaces import HasInnerState
import torch.nn.functional as F
from torch import nn
import copy
import triton
import triton.language as tl

logger = init_logger(__name__)

@triton.jit
def prefill_smooth_kernel(
    hidden_states_ptr, 
    filter_ptr,
    output_ptr,
    cu_seqlens_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_hidden_seq: tl.constexpr,
):
    pid_seq = tl.program_id(0)  #seq_id
    pid_pos = tl.program_id(1)  # token_id

    seq_start = tl.load(cu_seqlens_ptr + pid_seq)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1)

    i = seq_start + pid_pos
    if i >= seq_end:
        return

    offset_cur = i * stride_hidden_seq
    offset_prev = offset_cur - stride_hidden_seq

    pre_weight = tl.load(filter_ptr + tl.arange(0, num_heads)).reshape(num_heads, 1)
    cur_weight = tl.load(filter_ptr + num_heads + tl.arange(0, num_heads)).reshape(num_heads, 1)
    dim_offset = tl.arange(0, num_heads * head_dim)
    mask = pid_pos * num_heads * head_dim + dim_offset >= num_heads * head_dim
    prev_value = tl.load(hidden_states_ptr + offset_prev + dim_offset, mask=mask, other=0.0).reshape(num_heads, head_dim)
    cur_value = tl.load(hidden_states_ptr + offset_cur + dim_offset).reshape(num_heads, head_dim)

    smoothed_value = cur_value * cur_weight + prev_value * pre_weight
    smoothed_value = smoothed_value.reshape(num_heads * head_dim)
    
    output_offset = i * num_heads * head_dim
    tl.store(output_ptr + output_offset + dim_offset, smoothed_value) 

def prefill_smooth(
    hidden_states: torch.Tensor,  #[total_seq_len, num_heads, head_dim]
    cu_seqlens: torch.Tensor,  #seq_num+1:[0, seq1, seq1+seq2, ....]
    max_seq_len: int, 
    filter: torch.Tensor #[2, num_heads]
) -> torch.Tensor:
    total_seq_len, num_heads, head_dim = hidden_states.shape
    stride_hidden_seq = hidden_states.stride(0)
    seq_num = cu_seqlens.shape[0] - 1 
    output = torch.empty_like(hidden_states)
    prefill_smooth_kernel[(seq_num, max_seq_len)](
        hidden_states,
        filter,
        output,
        cu_seqlens,
        num_heads,
        head_dim,
        stride_hidden_seq
    )
    return output
    
@triton.jit
def decode_smooth_kernel(
    hidden_states_ptr,
    last_hidden_states_ptr,
    filter_ptr,
    output_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    pre_weight = tl.load(filter_ptr + head_id)  # ( 1)
    cur_weight = tl.load(filter_ptr + num_heads + head_id)  # ( 1)
    seq_offsets = seq_id * num_heads * head_dim + head_id * head_dim + tl.arange(0, head_dim)
    cur_value = tl.load(hidden_states_ptr + seq_offsets)  
    pre_value = tl.load(last_hidden_states_ptr + seq_offsets)
    smoothed_value = pre_value * pre_weight + cur_value * cur_weight 
    tl.store(output_ptr + seq_offsets, smoothed_value)
    
def decode_smooth(
    hidden_states: torch.Tensor, 
    last_hidden_states: torch.Tensor, 
    filter: torch.Tensor
) -> torch.Tensor:
    seq_num, num_heads, head_dim = hidden_states.shape
    output = torch.empty_like(hidden_states)
    last_hidden_states = last_hidden_states.contiguous()
    decode_smooth_kernel[(seq_num, num_heads)](
        hidden_states,
        last_hidden_states,
        filter,
        output,
        num_heads,
        head_dim,
    )
    return output
    
    
class LastKVCacheMannager:
    def __init__(
        self, 
        dtype: torch.dtype, 
        device: torch.device, 
        max_batch_size: int, 
        config: PretrainedConfig,
    ):
        
        self.device = device 
        num_layers = config.num_hidden_layers
        sliding_window_layers = config.sliding_window_layers
        num_swa_key_value_heads = config.num_swa_key_value_heads
        swa_key_value_head_dim = config.hidden_size // config.num_swa_attention_heads
        
        num_key_value_heads = config.num_key_value_heads
        key_value_head_dim = config.hidden_size // config.num_attention_heads
        
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        
        assert num_swa_key_value_heads % tensor_model_parallel_world_size == 0
        num_swa_key_value_heads = (num_swa_key_value_heads //
                             tensor_model_parallel_world_size)
        assert num_key_value_heads % tensor_model_parallel_world_size == 0
        num_key_value_heads = num_key_value_heads // tensor_model_parallel_world_size
        self.last_kv_caches = []
        for layer_id in range(num_layers):
            if layer_id in sliding_window_layers:
                self.last_kv_caches.append(torch.empty(size=(2, max_batch_size, num_swa_key_value_heads, swa_key_value_head_dim),
                                                       dtype=dtype,
                                                       device=device))
            else:
                self.last_kv_caches.append(torch.empty(size=(2,  max_batch_size, num_key_value_heads, key_value_head_dim),
                                                       dtype=dtype,
                                                       device=device))
        self.cache_indices_mapping: Dict[str, int] = {}
        self.free_cache_indices = list(range(max_batch_size))
    
    def _release_finished_requests(self, finished_seq_groups_req_ids: List[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.cache_indices_mapping:
                logger.info(f'req_id:{req_id}, is released, index:{self.cache_indices_mapping[req_id]}')
                index = self.cache_indices_mapping[req_id]
                self.free_cache_indices.append(index)
                self.cache_indices_mapping.pop(req_id)
            else:
                logger.warning(f'req_id:{req_id}, is not in cache_indices_mapping:{self.cache_indices_mapping}')
              
    def get_last_kv_tensors(self, request_ids_to_seq_ids, finished_requests_ids):
        self._release_finished_requests(finished_requests_ids)
        last_kv_indices = [0] * len(request_ids_to_seq_ids)
        for i, (req_id, _) in enumerate(request_ids_to_seq_ids.items()):
            if req_id in self.cache_indices_mapping:
                last_kv_indices[i] = self.cache_indices_mapping[req_id] 
            elif req_id in finished_requests_ids:
                #Warmup, donot allocate cache_indices.
                logger.warning(f'req_id:{req_id}, in finished_requests_ids:{finished_requests_ids}')
                last_kv_indices[i] = 0
            else:
                assert len(self.free_cache_indices) > 0
                index = self.free_cache_indices.pop()
                self.cache_indices_mapping[req_id] = index
                last_kv_indices[i] = index
        last_kv_indices = torch.tensor(last_kv_indices, dtype=torch.int32, device=self.device)
        return self.last_kv_caches, last_kv_indices

class BaiChuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BaiChuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        num_heads: int,
        num_kv_heads: int,
        is_swa: bool,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size(
        )
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tensor_model_parallel_world_size == 0
        self.num_kv_heads = (self.total_num_kv_heads //
                             tensor_model_parallel_world_size)
        self.is_swa = is_swa
        self.layer_id = layer_id
        self.sliding_window = self.config.interleaved_sliding_window
        self.rope_theta = self.config.rope_theta
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.cache_config = copy.copy(cache_config)
        if not self.is_swa:
            self.cache_config.sliding_window = None
        else:
            self.cache_config.sliding_window = self.sliding_window

        # pylint: disable=UNUSED-name
        self.W_pack = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.conv_window = 2
        self.K = nn.Parameter(
            torch.empty(1, 1, self.num_kv_heads, 1, self.conv_window,))
        self.V = nn.Parameter(
            torch.empty(1, 1, self.num_kv_heads, 1, self.conv_window,))
        
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=self.cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata: AttentionMetadata,
                last_kv_cache: Optional[torch.Tensor],
                last_kv_indices: Optional[torch.Tensor],) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        #Original dim = 5, but we need 2-D for inference.
        if self.K.dim() == 5 or self.V.dim() == 5: 
            self.K.data = self.K.data.view(self.num_kv_heads, self.conv_window,).transpose(0, 1).contiguous()
            self.V.data = self.V.data.view(self.num_kv_heads, self.conv_window,).transpose(0, 1).contiguous()
            
        num_prefills = attn_metadata.num_prefills
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        if attn_metadata.num_prefills > 0:
            assert attn_metadata.query_start_loc is not None, f'attn_metadata: {attn_metadata}'
            prefill_k, prefill_v = k[:num_prefill_tokens], v[:num_prefill_tokens]
            
            prefill_seq_lens = attn_metadata.seq_lens[:num_prefills]
            max_seqlen = max(prefill_seq_lens)
            prefill_query_start_loc = attn_metadata.query_start_loc[:num_prefills+1]
            last_token_copy_from_indices = prefill_query_start_loc[1:] - 1
            last_token_copy_to_indices = last_kv_indices[:num_prefills]
            last_kv_cache[0,last_token_copy_to_indices] = k[last_token_copy_from_indices].view(-1, self.num_kv_heads, self.head_dim)
            last_kv_cache[1,last_token_copy_to_indices] = v[last_token_copy_from_indices].view(-1, self.num_kv_heads, self.head_dim)
            
            smoothed_k = prefill_smooth(
                prefill_k.view(-1, self.num_kv_heads, self.head_dim),
                prefill_query_start_loc,
                max_seqlen,
                self.K
            )
            smoothed_v = prefill_smooth(
                prefill_v.view(-1, self.num_kv_heads, self.head_dim),
                prefill_query_start_loc,
                max_seqlen,
                self.V
            )
            k[:num_prefill_tokens] = smoothed_k.view_as(prefill_k)
            v[:num_prefill_tokens] = smoothed_v.view_as(prefill_v)
            
        if attn_metadata.num_decode_tokens > 0:            
            decode_k, decode_v = k[num_prefill_tokens:].clone(), v[num_prefill_tokens:].clone()
            last_token_indices = last_kv_indices[num_prefills:]
            last_k = last_kv_cache[0,last_token_indices]
            last_v = last_kv_cache[1,last_token_indices]
            
            smoothed_k = decode_smooth(
                decode_k.view(-1, self.num_kv_heads, self.head_dim),
                last_k,
                self.K,
            )
            smoothed_v = decode_smooth(
                decode_v.view(-1, self.num_kv_heads, self.head_dim),
                last_v,
                self.V,
            )
            
            #Set last k/v
            assert decode_k.shape[0] == last_token_indices.shape[0]
            last_kv_cache[0,last_token_indices] = decode_k.view(-1, self.num_kv_heads, self.head_dim)
            last_kv_cache[1,last_token_indices] = decode_v.view(-1, self.num_kv_heads, self.head_dim)
            
            k[num_prefill_tokens:] = smoothed_k.view_as(decode_k)
            v[num_prefill_tokens:] = smoothed_v.view_as(decode_v)
            
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class BaiChuanDecoderLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 num_heads: int,
                 num_kv_heads: int,
                 is_swa: bool,
                 layer_id: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaiChuanAttention(
            config=config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            is_swa=is_swa,
            layer_id=layer_id,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = BaiChuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        last_kv_cache: Optional[torch.Tensor],
        last_kv_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            last_kv_cache=last_kv_cache,
            last_kv_indices=last_kv_indices,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

@support_torch_compile
class BaiChuanModel(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        position_embedding: str = "ROPE",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # Note: donot support PP.
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([
            BaiChuanDecoderLayer(config,
                                 num_heads=self.get_num_heads(layer_id),
                                 num_kv_heads=self.get_num_kv_heads(layer_id),
                                 is_swa=layer_id
                                 in self.config.sliding_window_layers,
                                 layer_id=layer_id,
                                 cache_config=self.cache_config,
                                 quant_config=quant_config,
                                 prefix=f"{layer_id}.layers")
            for layer_id in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.last_kv_cache_manager: Optional[LastKVCacheManager] = None

    def get_num_heads(self, layer_idx: int):
        if layer_idx in self.config.sliding_window_layers:
            return self.config.num_swa_attention_heads
        return self.config.num_attention_heads

    def get_num_kv_heads(self, layer_idx: int):
        if layer_idx in self.config.sliding_window_layers:
            return self.config.num_swa_key_value_heads
        return self.config.num_key_value_heads
    
    def create_last_kv(self, num_blocks):
        pass 

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
    
        if self.last_kv_cache_manager is None:
            self.last_kv_cache_manager = LastKVCacheMannager(
                self.model_config.dtype, 
                input_ids.device, 
                self.max_num_seqs, 
                self.config,
            )
        #Ensure kwargs have request_ids_to_seq_ids and finished_requests_ids.
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        finished_requests_ids = kwargs["finished_requests_ids"]
        last_kv_caches, last_kv_indices = self.last_kv_cache_manager.get_last_kv_tensors(request_ids_to_seq_ids, 
                                                                                         finished_requests_ids,)

        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
                last_kv_caches[i],
                last_kv_indices,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

cnt = 0
class BaiChuanBaseForCausalLM(nn.Module, SupportsLoRA, SupportsPP,
                              HasInnerState):
    packed_modules_mapping = {
        "W_pack": ["W_pack"],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "W_pack",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".W_pack",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        position_embedding: str = "ROPE",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = BaiChuanModel(vllm_config=vllm_config,
                                   prefix=prefix,
                                   position_embedding=position_embedding)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors, **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name == "lm_head.weight":
                loaded_weight = torch.nn.functional.normalize(loaded_weight)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if "self_attn.K" in name or "self_attn.V" in name:
                    weight_loader = sharded_weight_loader(2)
                weight_loader(param, loaded_weight)


class BaiChuan5ForCausalLM(BaiChuanBaseForCausalLM):
    """Baichuan5 26B.
    NOTE: the class name has an upper case 'C'.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         position_embedding="ROPE")
