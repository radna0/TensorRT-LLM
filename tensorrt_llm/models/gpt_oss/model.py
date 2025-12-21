# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import numpy as np

from ..._utils import pad_vocab_size
from ...functional import Tensor, recv, send
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, PositionEmbeddingType, RmsNorm)
from ...mapping import Mapping
from ...module import Module
from ...parameter import Parameter
from ...quantization import QuantMode
from ..convert_utils import has_safetensors
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .convert import load_weights_from_hf_safetensors
from .config import GptOssConfig


class GptOssDecoderLayer(Module):

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        attention_quant_mode = config.quant_mode
        attn_name = f"transformer.layers.{layer_idx}.attention"
        if config.quantization.is_module_excluded_from_quantization(attn_name):
            attention_quant_mode = QuantMode.from_quant_algo(
                None, config.quantization.kv_cache_quant_algo)
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=attention_quant_mode,
            cp_group=config.mapping.cp_group,
            cp_size=config.mapping.cp_size,
            cp_rank=config.mapping.cp_rank)

        if config.quant_mode.has_kv_cache_quant():
            self.attention.kv_cache_scaling_factor = Parameter(shape=(1, ),
                                                               dtype='float32')
            self.attention.kv_cache_rcp_scaling_factor = Parameter(
                shape=(1, ), dtype='float32')

        self.attention.attention_sinks = Parameter(
            shape=(self.attention.num_attention_heads, ), dtype='float32')

        self.mlp = MOE(hidden_size=config.hidden_size,
                       ffn_hidden_size=config.intermediate_size,
                       hidden_act=config.hidden_act,
                       dtype=config.dtype,
                       bias=config.mlp_bias,
                       router_bias=True,
                       tp_group=config.mapping.tp_group,
                       tp_size=config.mapping.tp_size,
                       quant_mode=config.quant_mode,
                       moe_config=config.moe,
                       mapping=config.mapping)

        if self.mlp.hidden_act == 'swiglu_bias':
            experts_per_node = self.mlp.experts_per_node
            alpha = np.full((experts_per_node, ),
                            config.swiglu_alpha,
                            dtype=np.float32)
            beta = np.full((experts_per_node, ),
                           config.swiglu_beta,
                           dtype=np.float32)
            limit = np.full((experts_per_node, ),
                            config.swiglu_limit,
                            dtype=np.float32)
            self.mlp.swiglu_alpha.value = alpha
            self.mlp.swiglu_beta.value = beta
            self.mlp.swiglu_limit.value = limit

        self.post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                spec_decoding_params=None,
                mrope_params=None,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                next_layer_input_layernorm_args=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            mrope_params=mrope_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GptOssModel(Module):

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config

        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(GptOssDecoderLayer, config)

        if config.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                mrope_params=None):
        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.config.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states,
                                 self.config.mapping.prev_pp_rank())

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params,
                                    position_ids=position_ids,
                                    lora_params=lora_params,
                                    spec_decoding_params=spec_decoding_params,
                                    mrope_params=mrope_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states,
                                 self.config.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GptOssForCausalLM(DecoderModelForCausalLM):
    config_class = GptOssConfig

    def __init__(self, config: GptOssConfig):
        transformer = GptOssModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_config_or_dir = hf_model_or_dir.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = GptOssConfig.from_hugging_face(hf_config_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)

        if use_preloading:
            raise NotImplementedError(
                "GPT-OSS TRT loading expects a HF safetensors checkpoint directory."
            )

        if not has_safetensors(hf_model_dir):
            raise ValueError(
                f"GPT-OSS TRT loading expects safetensors weights in {hf_model_dir}."
            )

        weights = load_weights_from_hf_safetensors(hf_model_dir, config)
        model = cls(config)
        model.load(weights)
        return model
