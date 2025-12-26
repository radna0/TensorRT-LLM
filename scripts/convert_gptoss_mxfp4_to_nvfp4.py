#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from modelopt.torch.quantization.qtensor import NVFP4QTensor

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

DEFAULT_EXCLUDE_MODULES = [
    "block.*.attn.out",
    "block.*.mlp.gate",
    "block.*.attn.qkv",
    "embedding",
    "unembedding",
]


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _is_mxfp4_model(model_dir: Path) -> bool:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    cfg = _read_json(cfg_path)
    qcfg = cfg.get("quantization_config") or {}
    return qcfg.get("quant_method") == "mxfp4"


def _dequantize_mxfp4_blocks(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    rows_per_chunk: int,
) -> torch.Tensor:
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, groups, bytes_per_block = blocks.shape
    rows_total = math.prod(prefix_shape) * groups

    blocks = blocks.reshape(rows_total, bytes_per_block)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(
        rows_total,
        bytes_per_block * 2,
        dtype=dtype,
        device=blocks.device,
    )

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)

        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, groups, bytes_per_block * 2).view(
        *prefix_shape, groups * bytes_per_block * 2
    )
    return out.transpose(1, 2).contiguous()


def _convert_mxfp4_weight_to_nvfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    chunk_out: int,
    rows_per_chunk: int,
    block_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts, out_dim, groups, bytes_per_block = blocks.shape
    input_dim = groups * bytes_per_block * 2

    if input_dim % block_size != 0:
        raise ValueError(
            f"Input dim {input_dim} must be divisible by block size {block_size}."
        )

    amax = 0.0
    with torch.no_grad():
        for out_start in range(0, out_dim, chunk_out):
            out_end = min(out_start + chunk_out, out_dim)
            weight_chunk = _dequantize_mxfp4_blocks(
                blocks[:, out_start:out_end],
                scales[:, out_start:out_end],
                dtype=dtype,
                rows_per_chunk=rows_per_chunk,
            )
            chunk_max = float(weight_chunk.abs().max())
            if chunk_max > amax:
                amax = chunk_max
            del weight_chunk

    weight_scale_2 = torch.tensor(
        amax / (6.0 * 448.0),
        dtype=torch.float32,
        device=blocks.device,
    )
    if weight_scale_2.item() == 0.0:
        weight_scale_2 = torch.tensor(1.0, dtype=torch.float32, device=blocks.device)

    packed_weight = torch.empty(
        (num_experts, input_dim // 2, out_dim),
        dtype=torch.uint8,
        device=blocks.device,
    )
    weight_scale = torch.empty(
        (num_experts, input_dim // block_size, out_dim),
        dtype=torch.float8_e4m3fn,
        device=blocks.device,
    )

    with torch.no_grad():
        for out_start in range(0, out_dim, chunk_out):
            out_end = min(out_start + chunk_out, out_dim)
            weight_chunk = _dequantize_mxfp4_blocks(
                blocks[:, out_start:out_end],
                scales[:, out_start:out_end],
                dtype=dtype,
                rows_per_chunk=rows_per_chunk,
            )
            weight_chunk = weight_chunk.transpose(-2, -1).contiguous()

            qtensor, chunk_scale, _ = NVFP4QTensor.quantize(
                weight_chunk,
                block_size,
                weights_scaling_factor=None,
                weights_scaling_factor_2=weight_scale_2,
                try_tensorrt=False,
            )
            chunk_packed = qtensor._quantized_data.transpose(-2, -1).contiguous()
            chunk_scale = chunk_scale.transpose(-2, -1).contiguous()

            packed_weight[:, :, out_start:out_end] = chunk_packed
            weight_scale[:, :, out_start:out_end] = chunk_scale

            del weight_chunk, qtensor, chunk_scale, chunk_packed

    return packed_weight, weight_scale, weight_scale_2


def _collect_kv_only_keys(
    input_weight_map: dict, kv_cache_weight_map: dict
) -> dict[str, list[str]]:
    kv_only_keys = set(kv_cache_weight_map) - set(input_weight_map)
    keys_by_shard: dict[str, list[str]] = {}
    for key in kv_only_keys:
        shard = kv_cache_weight_map[key]
        keys_by_shard.setdefault(shard, []).append(key)
    return keys_by_shard


def _copy_metadata_files(model_dir: Path, output_dir: Path) -> None:
    for entry in model_dir.iterdir():
        if entry.is_dir():
            continue
        if entry.name.endswith(".safetensors"):
            continue
        if entry.name == "model.safetensors.index.json":
            continue
        if entry.name == "hf_quant_config.json":
            continue
        shutil.copy2(entry, output_dir / entry.name)


def _build_modelopt_quant_config(has_kv_cache: bool, block_size: int) -> dict:
    try:
        import modelopt
        modelopt_version = getattr(modelopt, "__version__", "unknown")
    except Exception:
        modelopt_version = "unknown"

    quant_config = {
        "producer": {
            "name": "modelopt",
            "version": modelopt_version,
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": block_size,
            "exclude_modules": DEFAULT_EXCLUDE_MODULES,
        },
    }
    if has_kv_cache:
        quant_config["quantization"]["kv_cache_quant_algo"] = "NVFP4"
    return quant_config


def _update_config_json(output_dir: Path, quant_config: dict) -> None:
    cfg_path = output_dir / "config.json"
    cfg = _read_json(cfg_path)
    try:
        from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format

        cfg["quantization_config"] = convert_hf_quant_config_format(quant_config)
    except Exception:
        cfg["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        }
    _write_json(cfg_path, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GPT-OSS MXFP4 experts to NVFP4 weights."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--kv-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-out", type=int, default=256)
    parser.add_argument("--rows-per-chunk", type=int, default=131072)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dequant-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    kv_cache_dir = args.kv_cache_dir.expanduser() if args.kv_cache_dir else None

    if not _is_mxfp4_model(model_dir):
        raise ValueError(
            f"{model_dir} does not look like an MXFP4 GPT-OSS checkpoint."
        )

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing model.safetensors.index.json in {model_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_metadata_files(model_dir, output_dir)

    input_index = _read_json(index_path)
    input_weight_map = input_index["weight_map"]
    shard_files = sorted(set(input_weight_map.values()))

    kv_keys_by_shard = {}
    kv_weight_map = {}
    if kv_cache_dir is not None:
        kv_index_path = kv_cache_dir / "model.safetensors.index.json"
        if not kv_index_path.is_file():
            raise FileNotFoundError(
                f"Missing model.safetensors.index.json in {kv_cache_dir}"
            )
        kv_index = _read_json(kv_index_path)
        kv_weight_map = kv_index["weight_map"]
        kv_keys_by_shard = _collect_kv_only_keys(input_weight_map, kv_weight_map)

    dtype = torch.bfloat16 if args.dequant_dtype == "bf16" else torch.float16
    total_size = 0
    output_weight_map: dict[str, str] = {}

    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        print(f"Processing {shard_name}...")
        state = load_file(shard_path)
        out_state = {}

        for key, value in state.items():
            if key.endswith("gate_up_proj_blocks"):
                scales_key = key.replace("_blocks", "_scales")
                if scales_key not in state:
                    raise KeyError(f"Missing {scales_key} for {key}")
                blocks = value
                scales = state[scales_key]
                print(f"  Converting {key} -> NVFP4")
                packed, w_scale, w_scale_2 = _convert_mxfp4_weight_to_nvfp4(
                    blocks,
                    scales,
                    chunk_out=args.chunk_out,
                    rows_per_chunk=args.rows_per_chunk,
                    block_size=args.block_size,
                    dtype=dtype,
                )
                base = key.replace("_blocks", "")
                out_state[base] = packed
                out_state[base + "_weight_scale"] = w_scale
                out_state[base + "_weight_scale_2"] = w_scale_2
            elif key.endswith("down_proj_blocks"):
                scales_key = key.replace("_blocks", "_scales")
                if scales_key not in state:
                    raise KeyError(f"Missing {scales_key} for {key}")
                blocks = value
                scales = state[scales_key]
                print(f"  Converting {key} -> NVFP4")
                packed, w_scale, w_scale_2 = _convert_mxfp4_weight_to_nvfp4(
                    blocks,
                    scales,
                    chunk_out=args.chunk_out,
                    rows_per_chunk=args.rows_per_chunk,
                    block_size=args.block_size,
                    dtype=dtype,
                )
                base = key.replace("_blocks", "")
                out_state[base] = packed
                out_state[base + "_weight_scale"] = w_scale
                out_state[base + "_weight_scale_2"] = w_scale_2
            elif key.endswith("gate_up_proj_scales") or key.endswith("down_proj_scales"):
                continue
            else:
                out_state[key] = value

        if kv_cache_dir is not None and shard_name in kv_keys_by_shard:
            kv_shard_path = kv_cache_dir / shard_name
            kv_state = load_file(kv_shard_path)
            for kv_key in kv_keys_by_shard[shard_name]:
                out_state[kv_key] = kv_state[kv_key]

        for key, value in out_state.items():
            output_weight_map[key] = shard_name
            if isinstance(value, torch.Tensor):
                total_size += value.numel() * value.element_size()

        save_file(out_state, output_dir / shard_name)
        del state, out_state

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    _write_json(output_dir / "model.safetensors.index.json", output_index)

    quant_config = _build_modelopt_quant_config(
        has_kv_cache=kv_cache_dir is not None,
        block_size=args.block_size,
    )
    _write_json(output_dir / "hf_quant_config.json", quant_config)
    _update_config_json(output_dir, quant_config)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
