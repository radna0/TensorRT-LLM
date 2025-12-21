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

import json
import os
import time
from typing import Optional

import safetensors
import torch

from ..._utils import pad_vocab_size, str_dtype_to_torch
from ...math_utils import pad_up
from ...logger import logger
from ..convert_utils import split
from .config import GptOssConfig


def _block_scale_interleave(scales: torch.Tensor) -> torch.Tensor:
    scales_cpu = scales.detach().cpu().contiguous()
    interleaved = torch.ops.trtllm.block_scale_interleave(
        scales_cpu.view(torch.uint8))
    rows = scales_cpu.shape[-2]
    cols = scales_cpu.shape[-1]
    rows_padded = pad_up(rows, 128)
    cols_padded = pad_up(cols, 4)
    if scales_cpu.dim() == 3:
        out_shape = (scales_cpu.shape[0], rows_padded, cols_padded)
    else:
        out_shape = (rows_padded, cols_padded)
    interleaved = interleaved.view(scales_cpu.dtype).reshape(out_shape)
    return interleaved


def _expert_slice(config: GptOssConfig) -> slice:
    if not config.mapping.has_moe_ep():
        return slice(None)
    experts_per_rank = config.moe.num_experts // config.mapping.moe_ep_size
    start = config.mapping.moe_ep_rank * experts_per_rank
    return slice(start, start + experts_per_rank)


def _expand_expert_scale(scale: Optional[torch.Tensor], count: int,
                         name: str) -> torch.Tensor:
    if scale is None:
        return torch.ones((count, ), dtype=torch.float32)
    scale = scale.to(torch.float32).reshape(-1)
    if scale.numel() == 1:
        return scale.repeat(count)
    if scale.numel() != count:
        raise ValueError(
            f"Expected {name} to have {count} elements, got {scale.numel()}.")
    return scale


def load_weights_from_hf_safetensors(model_dir: str, config: GptOssConfig):
    logger.info('Loading weights from Huggingface GPT-OSS safetensors...')
    tik = time.time()
    weights = {}

    model_dir = model_dir if model_dir.endswith("/") else model_dir + "/"
    safetensors_map = {}
    try:
        with open(model_dir + "model.safetensors.index.json", 'r') as fr:
            sharding_map = json.load(fr)
        shard_files = sorted(set(sharding_map['weight_map'].values()))
        shard_to_idx = {name: idx for idx, name in enumerate(shard_files)}
        for k, v in sharding_map['weight_map'].items():
            safetensors_map[k] = shard_to_idx[v]
    except FileNotFoundError:
        shard_files = []
        for name in os.listdir(model_dir):
            if name.endswith(".safetensors"):
                shard_files.append(name)
        shard_files.sort()

    safetensors_ptrs = [
        safetensors.safe_open(model_dir + shard_file,
                              framework="pt",
                              device="cpu") for shard_file in shard_files
    ]

    torch_dtype = str_dtype_to_torch(config.dtype)
    mapping = config.mapping

    kv_tp_size: Optional[int] = None
    kv_tp_rank: Optional[int] = None
    if config.num_key_value_heads < mapping.tp_size:
        kv_tp_size = config.num_key_value_heads
        kv_tp_rank = mapping.tp_rank * kv_tp_size // mapping.tp_size

    def load(key,
             dtype: Optional[torch.dtype] = None,
             tp_dim: int = -1,
             tp_size: Optional[int] = None,
             tp_rank: Optional[int] = None,
             expert_slice: Optional[slice] = None):
        ptr_idx = safetensors_map.get(key, 0)
        if key not in safetensors_ptrs[ptr_idx].keys():
            return None
        tensor_slice = safetensors_ptrs[ptr_idx].get_slice(key)
        tensor_shape = tensor_slice.get_shape()
        indices = [slice(None)] * len(tensor_shape)
        if expert_slice is not None and indices:
            indices[0] = expert_slice
        if tp_dim >= 0:
            if tp_size is None:
                tp_size = mapping.tp_size
            if tp_rank is None:
                tp_rank = mapping.tp_rank
            dim_size = tensor_shape[tp_dim]
            start = dim_size * tp_rank // tp_size
            end = dim_size * (tp_rank + 1) // tp_size
            indices[tp_dim] = slice(start, end)
        res = tensor_slice[tuple(indices)]
        if dtype is not None and res.dtype != dtype:
            res = res.to(dtype)
        return res.contiguous().detach().cpu()

    def load_qkv(prefix: str):
        q_weight = load(prefix + "q_proj.weight", torch_dtype, tp_dim=0)
        k_weight = load(prefix + "k_proj.weight",
                        torch_dtype,
                        tp_dim=0,
                        tp_size=kv_tp_size,
                        tp_rank=kv_tp_rank)
        v_weight = load(prefix + "v_proj.weight",
                        torch_dtype,
                        tp_dim=0,
                        tp_size=kv_tp_size,
                        tp_rank=kv_tp_rank)

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        q_bias = load(prefix + "q_proj.bias", torch_dtype, tp_dim=0)
        if q_bias is not None:
            k_bias = load(prefix + "k_proj.bias",
                          torch_dtype,
                          tp_dim=0,
                          tp_size=kv_tp_size,
                          tp_rank=kv_tp_rank)
            v_bias = load(prefix + "v_proj.bias",
                          torch_dtype,
                          tp_dim=0,
                          tp_size=kv_tp_size,
                          tp_rank=kv_tp_rank)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        else:
            qkv_bias = None
        return qkv_weight, qkv_bias

    if mapping.is_first_pp_rank():
        vocab = load("model.embed_tokens.weight", torch_dtype)
        if config.use_parallel_embedding:
            vocab = split(vocab, mapping.tp_size, mapping.tp_rank,
                          config.embedding_sharding_dim)
        weights['transformer.vocab_embedding.weight'] = vocab

    if mapping.is_last_pp_rank():
        pad_vocab = config.vocab_size % mapping.tp_size != 0
        vocab_size_padded = pad_vocab_size(config.vocab_size, mapping.tp_size)
        lm_head = load("lm_head.weight",
                       torch_dtype,
                       tp_dim=0,
                       tp_size=mapping.tp_size,
                       tp_rank=mapping.tp_rank) if not pad_vocab else load(
                           "lm_head.weight", torch_dtype)
        lm_head_loaded = lm_head is not None
        if lm_head is None:
            lm_head = load("model.embed_tokens.weight",
                           torch_dtype).clone()
        if pad_vocab:
            lm_head = torch.nn.functional.pad(
                lm_head, (0, 0, 0, vocab_size_padded - config.vocab_size),
                'constant', 0)
            lm_head = split(lm_head, mapping.tp_size, mapping.tp_rank, dim=0)
        elif not lm_head_loaded:
            lm_head = split(lm_head, mapping.tp_size, mapping.tp_rank, dim=0)
        weights['lm_head.weight'] = lm_head
        weights['transformer.ln_f.weight'] = load("model.norm.weight",
                                                  torch_dtype)

    layers_range = mapping.pp_layers(config.num_hidden_layers)
    expert_slice = _expert_slice(config)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = f"model.layers.{l}."
        tllm_prefix = f"transformer.layers.{layer_idx}."

        qkv_weight, qkv_bias = load_qkv(prefix + "self_attn.")
        weights[tllm_prefix + "attention.qkv.weight"] = qkv_weight
        if qkv_bias is not None:
            weights[tllm_prefix + "attention.qkv.bias"] = qkv_bias

        o_weight = load(prefix + "self_attn.o_proj.weight",
                        torch_dtype,
                        tp_dim=1)
        weights[tllm_prefix + "attention.dense.weight"] = o_weight
        o_bias = load(prefix + "self_attn.o_proj.bias", torch_dtype)
        if o_bias is not None:
            weights[tllm_prefix + "attention.dense.bias"] = o_bias

        sinks = load(prefix + "self_attn.sinks", torch.float32)
        if sinks is not None:
            sinks = split(sinks, mapping.tp_size, mapping.tp_rank, dim=0)
            weights[tllm_prefix + "attention.attention_sinks"] = sinks

        weights[tllm_prefix + "input_layernorm.weight"] = load(
            prefix + "input_layernorm.weight", torch_dtype)
        weights[tllm_prefix + "post_attention_layernorm.weight"] = load(
            prefix + "post_attention_layernorm.weight", torch_dtype)

        if config.moe.has_moe():
            router_weight = load(prefix + "mlp.router.weight", torch.float32)
            weights[tllm_prefix + "mlp.router.weight"] = router_weight
            router_bias = load(prefix + "mlp.router.bias", torch.float32)
            if router_bias is not None:
                weights[tllm_prefix + "mlp.router.bias"] = router_bias

            has_nvfp4 = config.quant_mode.has_nvfp4(
            ) or config.quant_mode.has_w4a8_nvfp4_fp8()
            if config.quant_mode.has_mxfp4():
                gate_up_blocks = load(prefix +
                                      "mlp.experts.gate_up_proj_blocks",
                                      None,
                                      expert_slice=expert_slice)
                gate_blocks = gate_up_blocks[:, ::2, :, :]
                up_blocks = gate_up_blocks[:, 1::2, :, :]
                gate_up_blocks = torch.cat([up_blocks, gate_blocks], dim=1)
                gate_up_blocks = gate_up_blocks.flatten(-2, -1)
                if mapping.has_moe_tp():
                    gate_up_blocks = split(gate_up_blocks, mapping.moe_tp_size,
                                           mapping.moe_tp_rank, dim=1)
                weights[tllm_prefix + "mlp.fc.weight"] = gate_up_blocks

                down_blocks = load(prefix + "mlp.experts.down_proj_blocks",
                                   None,
                                   expert_slice=expert_slice)
                down_blocks = down_blocks.flatten(-2, -1)
                if mapping.has_moe_tp():
                    down_blocks = split(down_blocks, mapping.moe_tp_size,
                                        mapping.moe_tp_rank, dim=2)
                weights[tllm_prefix + "mlp.proj.weight"] = down_blocks

                gate_up_bias = load(prefix + "mlp.experts.gate_up_proj_bias",
                                    torch_dtype,
                                    expert_slice=expert_slice)
                if gate_up_bias is not None:
                    gate_bias = gate_up_bias[:, ::2]
                    up_bias = gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([up_bias, gate_bias], dim=1)
                    if mapping.has_moe_tp():
                        gate_up_bias = split(gate_up_bias,
                                             mapping.moe_tp_size,
                                             mapping.moe_tp_rank,
                                             dim=1)
                    weights[tllm_prefix + "mlp.fc.bias"] = gate_up_bias

                down_bias = load(prefix + "mlp.experts.down_proj_bias",
                                 torch_dtype,
                                 expert_slice=expert_slice)
                if down_bias is not None:
                    weights[tllm_prefix + "mlp.proj.bias"] = down_bias

                gate_up_scales = load(
                    prefix + "mlp.experts.gate_up_proj_scales",
                    None,
                    expert_slice=expert_slice)
                gate_scales = gate_up_scales[:, ::2, :]
                up_scales = gate_up_scales[:, 1::2, :]
                gate_up_scales = torch.cat([up_scales, gate_scales], dim=1)
                if mapping.has_moe_tp():
                    gate_up_scales = split(gate_up_scales,
                                           mapping.moe_tp_size,
                                           mapping.moe_tp_rank,
                                           dim=1)
                gate_up_scales_interleaved = _block_scale_interleave(
                    gate_up_scales)
                weights[tllm_prefix +
                        "mlp.fc.weights_block_scaling_factor"] = gate_up_scales
                weights[
                    tllm_prefix +
                    "mlp.fc.weights_block_scaling_factor_interleaved"] = gate_up_scales_interleaved

                down_scales = load(prefix + "mlp.experts.down_proj_scales",
                                   None,
                                   expert_slice=expert_slice)
                if mapping.has_moe_tp():
                    down_scales = split(down_scales, mapping.moe_tp_size,
                                        mapping.moe_tp_rank, dim=2)
                down_scales_interleaved = _block_scale_interleave(down_scales)
                weights[tllm_prefix +
                        "mlp.proj.weights_block_scaling_factor"] = down_scales
                weights[
                    tllm_prefix +
                    "mlp.proj.weights_block_scaling_factor_interleaved"] = down_scales_interleaved

                experts_per_node = gate_up_blocks.shape[0]
                one_scalar = torch.tensor([1.0], dtype=torch.float32)
                weights[
                    tllm_prefix +
                    "mlp.fc.activation_global_scaling_factor"] = one_scalar
                weights[
                    tllm_prefix +
                    "mlp.proj.activation_global_scaling_factor"] = one_scalar
                weights[tllm_prefix + "mlp.fc.alpha"] = torch.ones(
                    (experts_per_node, ), dtype=torch.float32)
                weights[tllm_prefix + "mlp.proj.alpha"] = torch.ones(
                    (experts_per_node, ), dtype=torch.float32)
            elif has_nvfp4:
                gate_up_weight = load(prefix + "mlp.experts.gate_up_proj",
                                      None,
                                      expert_slice=expert_slice)
                if gate_up_weight is None:
                    raise ValueError(
                        "Missing gate_up_proj weights for NVFP4 GPT-OSS.")
                gate_weight = gate_up_weight[:, :, ::2]
                up_weight = gate_up_weight[:, :, 1::2]
                gate_up_weight = torch.cat([up_weight, gate_weight], dim=2)
                gate_up_weight = gate_up_weight.transpose(1, 2).contiguous()
                if mapping.has_moe_tp():
                    gate_up_weight = split(gate_up_weight,
                                           mapping.moe_tp_size,
                                           mapping.moe_tp_rank,
                                           dim=1)
                weights[tllm_prefix + "mlp.fc.weight"] = gate_up_weight

                down_weight = load(prefix + "mlp.experts.down_proj",
                                   None,
                                   expert_slice=expert_slice)
                if down_weight is None:
                    raise ValueError(
                        "Missing down_proj weights for NVFP4 GPT-OSS.")
                down_weight = down_weight.transpose(1, 2).contiguous()
                if mapping.has_moe_tp():
                    down_weight = split(down_weight, mapping.moe_tp_size,
                                        mapping.moe_tp_rank, dim=2)
                weights[tllm_prefix + "mlp.proj.weight"] = down_weight

                gate_up_bias = load(prefix + "mlp.experts.gate_up_proj_bias",
                                    torch_dtype,
                                    expert_slice=expert_slice)
                if gate_up_bias is not None:
                    gate_bias = gate_up_bias[:, ::2]
                    up_bias = gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([up_bias, gate_bias], dim=1)
                    if mapping.has_moe_tp():
                        gate_up_bias = split(gate_up_bias,
                                             mapping.moe_tp_size,
                                             mapping.moe_tp_rank,
                                             dim=1)
                    weights[tllm_prefix + "mlp.fc.bias"] = gate_up_bias

                down_bias = load(prefix + "mlp.experts.down_proj_bias",
                                 torch_dtype,
                                 expert_slice=expert_slice)
                if down_bias is not None:
                    weights[tllm_prefix + "mlp.proj.bias"] = down_bias

                gate_up_scales = load(
                    prefix + "mlp.experts.gate_up_proj_weight_scale",
                    None,
                    expert_slice=expert_slice)
                if gate_up_scales is None:
                    raise ValueError(
                        "Missing gate_up_proj_weight_scale for NVFP4 GPT-OSS."
                    )
                gate_scales = gate_up_scales[:, :, ::2]
                up_scales = gate_up_scales[:, :, 1::2]
                gate_up_scales = torch.cat([up_scales, gate_scales], dim=2)
                gate_up_scales = gate_up_scales.transpose(1,
                                                          2).contiguous()
                if mapping.has_moe_tp():
                    gate_up_scales = split(gate_up_scales,
                                           mapping.moe_tp_size,
                                           mapping.moe_tp_rank,
                                           dim=1)
                gate_up_scales_interleaved = _block_scale_interleave(
                    gate_up_scales)
                weights[tllm_prefix +
                        "mlp.fc.weights_block_scaling_factor"] = gate_up_scales
                weights[
                    tllm_prefix +
                    "mlp.fc.weights_block_scaling_factor_interleaved"] = gate_up_scales_interleaved

                down_scales = load(
                    prefix + "mlp.experts.down_proj_weight_scale",
                    None,
                    expert_slice=expert_slice)
                if down_scales is None:
                    raise ValueError(
                        "Missing down_proj_weight_scale for NVFP4 GPT-OSS.")
                down_scales = down_scales.transpose(1, 2).contiguous()
                if mapping.has_moe_tp():
                    down_scales = split(down_scales, mapping.moe_tp_size,
                                        mapping.moe_tp_rank, dim=2)
                down_scales_interleaved = _block_scale_interleave(down_scales)
                weights[tllm_prefix +
                        "mlp.proj.weights_block_scaling_factor"] = down_scales
                weights[
                    tllm_prefix +
                    "mlp.proj.weights_block_scaling_factor_interleaved"] = down_scales_interleaved

                experts_per_node = gate_up_weight.shape[0]
                one_scalar = torch.tensor([1.0], dtype=torch.float32)
                weights[
                    tllm_prefix +
                    "mlp.fc.activation_global_scaling_factor"] = one_scalar
                weights[
                    tllm_prefix +
                    "mlp.proj.activation_global_scaling_factor"] = one_scalar

                gate_up_scale_2 = load(
                    prefix + "mlp.experts.gate_up_proj_weight_scale_2",
                    torch.float32,
                    expert_slice=expert_slice)
                down_scale_2 = load(
                    prefix + "mlp.experts.down_proj_weight_scale_2",
                    torch.float32,
                    expert_slice=expert_slice)
                weights[tllm_prefix +
                        "mlp.fc.alpha"] = _expand_expert_scale(
                            gate_up_scale_2, experts_per_node,
                            "gate_up_proj_weight_scale_2")
                weights[tllm_prefix +
                        "mlp.proj.alpha"] = _expand_expert_scale(
                            down_scale_2, experts_per_node,
                            "down_proj_weight_scale_2")
            else:
                raise ValueError(
                    "GPT-OSS MoE weights require MXFP4 or NVFP4 quantization; "
                    "set quant_config.quant_algo appropriately.")

        if config.quant_mode.has_kv_cache_quant():
            kv_scale = load(prefix + "self_attn.k_proj.k_scale",
                            torch.float32)
            if kv_scale is not None:
                v_scale = load(prefix + "self_attn.v_proj.v_scale",
                               torch.float32)
                if v_scale is None:
                    raise ValueError(
                        f"Missing v_proj.v_scale for {prefix} when loading KV cache quant scales."
                    )
                kv_scale = torch.max(kv_scale, v_scale)
                if kv_scale.numel() != 1:
                    kv_scale = kv_scale.max().reshape(1)
                else:
                    kv_scale = kv_scale.reshape(1)
            else:
                kv_scale = torch.tensor([1.0], dtype=torch.float32)
            weights[tllm_prefix +
                    "attention.kv_cache_scaling_factor"] = kv_scale
            weights[tllm_prefix +
                    "attention.kv_cache_rcp_scaling_factor"] = torch.reciprocal(
                        kv_scale)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights
