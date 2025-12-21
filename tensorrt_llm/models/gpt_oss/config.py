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

from typing import List, Optional, Union

from ...layers import MoeConfig
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


def _extend_exclude_modules(quant_config: QuantConfig,
                            hf_quant_config: Optional[dict]) -> None:
    if quant_config is None:
        return
    exclude_modules = list(quant_config.exclude_modules or [])
    if isinstance(hf_quant_config, dict):
        modules_to_not_convert = hf_quant_config.get("modules_to_not_convert")
        if isinstance(modules_to_not_convert, list):
            exclude_modules.extend(modules_to_not_convert)
    if not exclude_modules:
        return
    trt_exclude = []
    for name in exclude_modules:
        if not isinstance(name, str):
            continue
        lowered = name.lower()
        if "self_attn" in lowered or "attn" in lowered:
            trt_exclude.append("transformer.layers.*.attention")
        if "attn.qkv" in lowered or "qkv" in lowered or "q_proj" in lowered:
            trt_exclude.append("transformer.layers.*.attention.qkv")
        if "attn.out" in lowered or "o_proj" in lowered:
            trt_exclude.append("transformer.layers.*.attention.dense")
        if "mlp.router" in lowered or "router" in lowered:
            trt_exclude.append("transformer.layers.*.mlp.router")
        if "embed_tokens" in lowered or "embedding" in lowered:
            trt_exclude.append("transformer.vocab_embedding")
        if "lm_head" in lowered or "unembedding" in lowered:
            trt_exclude.append("lm_head")
    quant_config.exclude_modules = sorted(
        set(exclude_modules + trt_exclude))


class GptOssConfig(PretrainedConfig):

    def __init__(
        self,
        *,
        attn_bias: bool = False,
        mlp_bias: bool = True,
        rotary_base: float = 10000.0,
        rotary_scaling: Optional[dict] = None,
        moe: Optional[Union[MoeConfig, dict]] = None,
        num_local_experts: int = 0,
        experts_per_token: int = 0,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
        attention_window_sizes: Optional[List[int]] = None,
        layer_types: Optional[List[str]] = None,
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: float = 7.0,
        **kwargs,
    ):
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token
        self.num_experts_per_tok = experts_per_token
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.attention_window_sizes = attention_window_sizes
        self.layer_types = layer_types
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        if moe is None:
            moe = MoeConfig(
                num_experts=num_local_experts,
                top_k=experts_per_token,
                normalization_mode=MoeConfig.ExpertScaleNormalizationMode.
                RENORMALIZE)
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output['attn_bias'] = self.attn_bias
        output['mlp_bias'] = self.mlp_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['moe'] = self.moe.to_dict()
        output['num_local_experts'] = self.num_local_experts
        output['experts_per_token'] = self.experts_per_token
        output['num_experts_per_tok'] = self.num_experts_per_tok
        output['head_dim'] = self.head_dim
        output['sliding_window'] = self.sliding_window
        output['attention_window_sizes'] = self.attention_window_sizes
        output['layer_types'] = self.layer_types
        output['swiglu_alpha'] = self.swiglu_alpha
        output['swiglu_beta'] = self.swiglu_beta
        output['swiglu_limit'] = self.swiglu_limit
        runtime_defaults = self.runtime_defaults
        if runtime_defaults is not None:
            output['runtime_defaults'] = {
                'max_attention_window': runtime_defaults.max_attention_window,
                'sink_token_length': runtime_defaults.sink_token_length,
            }
        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir,
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        trust_remote_code = kwargs.pop('trust_remote_code', True)
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_or_dir, trust_remote_code=trust_remote_code)

        head_dim = getattr(hf_config, 'head_dim',
                           hf_config.hidden_size //
                           hf_config.num_attention_heads)
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads',
                                      hf_config.num_attention_heads)
        experts_per_token = getattr(hf_config, 'experts_per_token', None)
        if experts_per_token is None:
            experts_per_token = getattr(hf_config, 'num_experts_per_tok', 0)
        num_local_experts = getattr(hf_config, 'num_local_experts', 0)
        sliding_window = getattr(hf_config, 'sliding_window', None)
        hf_layer_types = getattr(hf_config, 'layer_types', None)

        attention_window_sizes = None
        if hf_layer_types and sliding_window is not None:
            attention_window_sizes = []
            for layer_type in hf_layer_types:
                if layer_type == 'sliding_attention':
                    attention_window_sizes.append(int(sliding_window))
                else:
                    attention_window_sizes.append(
                        int(hf_config.max_position_embeddings))
        elif sliding_window is not None:
            attention_window_sizes = [int(sliding_window)]

        runtime_defaults = None
        if attention_window_sizes is not None:
            runtime_defaults = {
                'max_attention_window': attention_window_sizes,
            }

        moe_config = MoeConfig(
            num_experts=num_local_experts,
            top_k=experts_per_token,
            normalization_mode=MoeConfig.ExpertScaleNormalizationMode.
            RENORMALIZE)
        moe_config.validate()

        hf_quant_config = getattr(hf_config, 'quantization_config', None)
        if quant_config is None:
            if isinstance(hf_quant_config, dict):
                quant_method = hf_quant_config.get('quant_method', None)
                quant_algo_name = hf_quant_config.get(
                    'quant_algo', None) or hf_quant_config.get(
                        'quantization', {}).get('quant_algo', None)

                def parse_quant_algo(name: Optional[str]) -> Optional[QuantAlgo]:
                    if not name:
                        return None
                    if isinstance(name, QuantAlgo):
                        return name
                    if not isinstance(name, str):
                        return None
                    key = name.upper().replace('-', '_')
                    return QuantAlgo.__members__.get(key, None)

                if quant_method == 'mxfp4':
                    quant_algo = parse_quant_algo(quant_algo_name)
                    if quant_algo is None:
                        quant_algo = QuantAlgo.W4A16_MXFP4
                    quant_config = QuantConfig(quant_algo=quant_algo)
                elif quant_method == 'modelopt':
                    quant_algo = parse_quant_algo(quant_algo_name)
                    kv_cache_scheme = hf_quant_config.get(
                        'kv_cache_scheme',
                        hf_quant_config.get('kv_cache_quant_algo', None))
                    kv_cache_algo = parse_quant_algo(kv_cache_scheme)
                    if quant_algo is not None or kv_cache_algo is not None:
                        quant_config = QuantConfig(
                            quant_algo=quant_algo,
                            kv_cache_quant_algo=kv_cache_algo)

        _extend_exclude_modules(quant_config, hf_quant_config)

        source_dtype = getattr(hf_config, 'torch_dtype', None)
        if dtype == 'auto':
            source_dtype = 'bfloat16'
        dtype = infer_dtype(dtype, source_dtype)
        tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        layer_types = ['attention'] * hf_config.num_hidden_layers

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            vocab_size=hf_config.vocab_size,
            position_embedding_type='rope_gpt_neox',
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act='swiglu_bias',
            norm_epsilon=hf_config.rms_norm_eps,
            rotary_base=hf_config.rope_theta,
            rotary_scaling=getattr(hf_config, 'rope_scaling', None),
            attn_bias=getattr(hf_config, 'attention_bias',
                              getattr(hf_config, 'attn_bias', False)),
            mlp_bias=True,
            head_dim=head_dim,
            head_size=head_dim,
            num_local_experts=num_local_experts,
            experts_per_token=experts_per_token,
            sliding_window=sliding_window,
            attention_window_sizes=attention_window_sizes,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            swiglu_limit=getattr(hf_config, 'swiglu_limit', 7.0),
            layer_types=layer_types,
            tie_word_embeddings=tie_word_embeddings,
            mapping=mapping,
            quantization=quant_config,
            runtime_defaults=runtime_defaults,
            **kwargs,
        )
