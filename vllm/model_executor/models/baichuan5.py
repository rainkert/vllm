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

from .configuration_baichuan import BaiChuan5Config
from .interfaces import HasInnerState
import torch.nn.functional as F
from torch import nn
import copy


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


def custom_convolution(U, K_expanded):
    """
    U: 输入矩阵, 形状为 (bs, seq, h, d)
    K: 卷积核, 形状为 (w, h)
    返回: 输出矩阵 V, 形状为 (bs, seq, h, d)
    """
    bs, seq, h, d = U.shape
    _, _, h, _, w = K_expanded.shape
    padding = (w - 1, 0)
    U_padded = F.pad(U, (0, 0, 0, 0, *padding))  # 形状变为 (bs, seq+w-1, h, d)
    U_unfolded = U_padded.unfold(1, w, 1)  # 形状变为 (bs, seq+w-1, h, d, w)
    #K_expanded = K.unsqueeze(0).unsqueeze(0).unsqueeze(-2)  # 形状变为 (1, 1, h, 1, w)
    V_unfolded = U_unfolded * K_expanded  # 形状保持为 (bs, seq, h, d, w)
    V = V_unfolded.sum(dim=-1)  # 形状变为 (bs, seq, h, d)
    return V


class BaiChuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: BaiChuan5Config,
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

        # pylint: disable=invalid-name
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
            torch.softmax(torch.randn(
                (1, 1, self.num_kv_heads, 1, self.conv_window)),
                          dim=-1))
        self.V = nn.Parameter(
            torch.softmax(torch.randn(
                (1, 1, self.num_kv_heads, 1, self.conv_window)),
                          dim=-1))
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=self.cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")
        self.last_block_tables = None
        self.last_k = None
        self.last_v = None

    def prefill_set_last_kv(self, k: torch.Tensor, v: torch.Tensor,
                            last_kv_cache: torch.Tensor,
                            attn_metadata: AttentionMetadata):
        batch_size = attn_metadata.num_prefills
        token_start_idx = 0
        for seq_idx in range(batch_size):
            seq_len = attn_metadata.seq_lens[seq_idx]
            page_idx = attn_metadata.slot_mapping[
                attn_metadata.seq_start_loc[seq_idx + 1] - 1] // 64
            last_kv_cache[0][page_idx] = k[:, token_start_idx + seq_len - 1]
            last_kv_cache[1][page_idx] = v[:, token_start_idx + seq_len - 1]
            token_start_idx += seq_len

    def decode_set_last_kv(self, k: torch.Tensor, v: torch.Tensor,
                           last_kv_cache: torch.Tensor,
                           attn_metadata: AttentionMetadata):
        batch_size = attn_metadata.num_decode_tokens
        for seq_idx in range(batch_size):
            seq_len = attn_metadata.seq_lens[seq_idx]
            page_idx = attn_metadata.block_tables[seq_idx][seq_len // 64]
            last_kv_cache[0][page_idx] = k[:, seq_idx]
            last_kv_cache[1][page_idx] = v[:, seq_idx]

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata: AttentionMetadata,
                last_kv_cache: Optional[torch.Tensor]) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k = k.view(1, -1, self.num_kv_heads, self.head_dim)
        v = v.view(1, -1, self.num_kv_heads, self.head_dim)
        if attn_metadata.num_prefills > 0 and attn_metadata.num_decode_tokens == 0:
            batch_size = attn_metadata.num_prefills
            if kv_cache is not None and kv_cache.numel() != 0:
                self.prefill_set_last_kv(
                    k,
                    v,
                    last_kv_cache,
                    attn_metadata,
                )
            token_start_idx = 0
            for seq_idx in range(batch_size):
                seq_len = attn_metadata.seq_lens[seq_idx]
                k[:, token_start_idx:token_start_idx +
                  seq_len, :, :] = custom_convolution(
                      k[:, token_start_idx:token_start_idx + seq_len, :, :],
                      self.K)
                v[:, token_start_idx:token_start_idx +
                  seq_len, :, :] = custom_convolution(
                      v[:, token_start_idx:token_start_idx + seq_len, :, :],
                      self.V)
                token_start_idx += seq_len

        elif attn_metadata.num_prefill_tokens == 0 and attn_metadata.num_decode_tokens > 0:
            batch_size = attn_metadata.num_decode_tokens
            k_temp = k.clone()
            v_temp = v.clone()
            token_idx = 0
            for seq_idx in range(batch_size):

                seq_len = attn_metadata.seq_lens[seq_idx]
                page_idx = attn_metadata.block_tables[seq_idx][seq_len // 64]
                k[:,
                  token_idx, :, :] = self.K[0, 0, :, 0, :1] * last_kv_cache[0][
                      page_idx, :, :] + self.K[0, 0, :, 0,
                                               1:] * k[:, token_idx, :, :]
                v[:,
                  token_idx, :, :] = self.V[0, 0, :, 0, :1] * last_kv_cache[1][
                      page_idx, :, :] + self.V[0, 0, :, 0,
                                               1:] * v[:, token_idx, :, :]
                token_idx += 1
            self.decode_set_last_kv(k_temp, v_temp, last_kv_cache,
                                    attn_metadata)

        k = k.view(-1, self.num_kv_heads * self.head_dim)
        v = v.view(-1, self.num_kv_heads * self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class BaiChuanDecoderLayer(nn.Module):

    def __init__(self,
                 config: BaiChuan5Config,
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
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([
            BaiChuanDecoderLayer(config,
                                 num_heads=self.get_num_heads(layer_id),
                                 num_kv_heads=self.get_num_kv_heads(layer_id),
                                 is_swa=layer_id
                                 in self.config.sliding_window_layers,
                                 layer_id=layer_id,
                                 cache_config=cache_config,
                                 quant_config=quant_config,
                                 prefix=f"{layer_id}.layers")
            for layer_id in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.last_kv_caches = [None] * config.num_hidden_layers

    def get_num_heads(self, layer_idx: int):
        if layer_idx in self.config.sliding_window_layers:
            return self.config.num_swa_attention_heads
        return self.config.num_attention_heads

    def get_num_kv_heads(self, layer_idx: int):
        if layer_idx in self.config.sliding_window_layers:
            return self.config.num_swa_key_value_heads
        return self.config.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if kv_caches[0] is not None and kv_caches[0].numel() != 0 and (
                self.last_kv_caches[0] is None
                or self.last_kv_caches[0].size(1) != kv_caches[0].size(1)):
            for i in range(len(kv_caches)):
                self.last_kv_caches[i] = torch.empty(
                    kv_caches[i].size(0),
                    kv_caches[i].size(1),
                    kv_caches[i].size(3),
                    kv_caches[i].size(4),
                    dtype=kv_caches[i].dtype,
                    device=kv_caches[i].device)

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
                self.last_kv_caches[i],
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BaiChuanNormHead(nn.Module):

    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        norm_weight = nn.functional.normalize(self.weight)
        return nn.functional.linear(hidden_states, norm_weight)


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
                                   attn_metadata, intermediate_tensors)
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
            # (param_name, shard_name, shard_id)
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
