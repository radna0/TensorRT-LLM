#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"
QFORMAT="${QFORMAT:-${3:-}}"
QFORMAT_SET="0"
if [[ -n "${QFORMAT}" ]]; then
  QFORMAT_SET="1"
else
  QFORMAT="full_prec"
fi

if [[ -z "${MODEL_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: MODEL_DIR=... OUTPUT_DIR=... $0 [model_dir] [output_dir] [qformat]" >&2
  exit 1
fi

CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail}"
CALIB_SIZE="${CALIB_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-512}"
TOKENIZER_MAX_SEQ_LENGTH="${TOKENIZER_MAX_SEQ_LENGTH:-2048}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KV_CACHE_DTYPE_SET="0"
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  KV_CACHE_DTYPE_SET="1"
fi
if [[ -z "${KV_CACHE_DTYPE}" ]]; then
  KV_CACHE_DTYPE="$(python - <<'PY'
import os

kv = "fp8"
force = os.environ.get("FORCE_NVFP4", "").strip()

try:
    import torch
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 10:
            try:
                import modelopt.torch.quantization as mtq
                if hasattr(mtq, "NVFP4_KV_CFG"):
                    kv = "nvfp4"
            except Exception:
                pass
except Exception:
    pass

if force:
    kv = "nvfp4"

print(kv)
PY
)"
fi

IS_MXFP4="$(MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import json
import pathlib
import os

model_dir = pathlib.Path(os.environ.get("MODEL_DIR", "")).expanduser()
cfg_path = model_dir / "config.json"
if not cfg_path.is_file():
    print("0")
    raise SystemExit(0)
try:
    data = json.loads(cfg_path.read_text())
except Exception:
    print("0")
    raise SystemExit(0)

qcfg = data.get("quantization_config") or {}
print("1" if qcfg.get("quant_method") == "mxfp4" else "0")
PY
)"

if [[ "${IS_MXFP4}" == "1" ]]; then
  if [[ "${QFORMAT_SET}" == "0" ]]; then
    QFORMAT="nvfp4"
    echo "Detected MXFP4 pre-quantized GPT-OSS; defaulting qformat=nvfp4."
  elif [[ "${QFORMAT}" != "nvfp4" ]]; then
    echo "Warning: MXFP4 model detected but qformat=${QFORMAT}. For NVFP4 conversion, set QFORMAT=nvfp4."
  fi
  if [[ "${KV_CACHE_DTYPE_SET}" == "0" ]]; then
    KV_CACHE_DTYPE="nvfp4"
    echo "Detected MXFP4 pre-quantized GPT-OSS; defaulting kv_cache_dtype=nvfp4."
  elif [[ "${KV_CACHE_DTYPE}" != "nvfp4" ]]; then
    echo "Warning: MXFP4 model detected but kv_cache_dtype=${KV_CACHE_DTYPE}. For NVFP4 conversion, set KV_CACHE_DTYPE=nvfp4."
  fi
fi

echo "Using qformat=${QFORMAT}"
echo "Using kv_cache_dtype=${KV_CACHE_DTYPE}"

if [[ "${IS_MXFP4}" == "1" && "${QFORMAT}" == "nvfp4" ]]; then
  CONVERT_CHUNK_OUT="${MXFP4_TO_NVFP4_CHUNK_OUT:-256}"
  CONVERT_ROWS_PER_CHUNK="${MXFP4_TO_NVFP4_ROWS_PER_CHUNK:-131072}"
  CONVERT_BLOCK_SIZE="${MXFP4_TO_NVFP4_BLOCK_SIZE:-16}"
  KV_CACHE_TMP=""

  if [[ "${KV_CACHE_DTYPE}" == "nvfp4" ]]; then
    KV_CACHE_TMP="${OUTPUT_DIR}_kv_cache_tmp"
    echo "Calibrating KV cache (nvfp4) from MXFP4 checkpoint -> ${KV_CACHE_TMP}"
    python examples/quantization/quantize.py \
      --model_dir "${MODEL_DIR}" \
      --qformat "full_prec" \
      --kv_cache_dtype "${KV_CACHE_DTYPE}" \
      --calib_dataset "${CALIB_DATASET}" \
      --calib_size "${CALIB_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --calib_max_seq_length "${CALIB_MAX_SEQ_LENGTH}" \
      --tokenizer_max_seq_length "${TOKENIZER_MAX_SEQ_LENGTH}" \
      --tp_size "${TP_SIZE}" \
      --pp_size "${PP_SIZE}" \
      --cp_size "${CP_SIZE}" \
      --device "${DEVICE}" \
      --device_map "${DEVICE_MAP}" \
      --output_dir "${KV_CACHE_TMP}"

    echo "Converting MXFP4 experts to NVFP4 and merging KV cache scales..."
    python scripts/convert_gptoss_mxfp4_to_nvfp4.py \
      "${MODEL_DIR}" \
      "${OUTPUT_DIR}" \
      --kv-cache-dir "${KV_CACHE_TMP}" \
      --chunk-out "${CONVERT_CHUNK_OUT}" \
      --rows-per-chunk "${CONVERT_ROWS_PER_CHUNK}" \
      --block-size "${CONVERT_BLOCK_SIZE}"
  else
    echo "Converting MXFP4 experts to NVFP4 (no KV cache quantization)..."
    python scripts/convert_gptoss_mxfp4_to_nvfp4.py \
      "${MODEL_DIR}" \
      "${OUTPUT_DIR}" \
      --chunk-out "${CONVERT_CHUNK_OUT}" \
      --rows-per-chunk "${CONVERT_ROWS_PER_CHUNK}" \
      --block-size "${CONVERT_BLOCK_SIZE}"
  fi

  if [[ -n "${KV_CACHE_TMP}" ]]; then
    echo "Done. If you don't need the temporary KV cache checkpoint, remove: ${KV_CACHE_TMP}"
  fi
  exit 0
fi

python examples/quantization/quantize.py \
  --model_dir "${MODEL_DIR}" \
  --qformat "${QFORMAT}" \
  --kv_cache_dtype "${KV_CACHE_DTYPE}" \
  --calib_dataset "${CALIB_DATASET}" \
  --calib_size "${CALIB_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --calib_max_seq_length "${CALIB_MAX_SEQ_LENGTH}" \
  --tokenizer_max_seq_length "${TOKENIZER_MAX_SEQ_LENGTH}" \
  --tp_size "${TP_SIZE}" \
  --pp_size "${PP_SIZE}" \
  --cp_size "${CP_SIZE}" \
  --device "${DEVICE}" \
  --device_map "${DEVICE_MAP}" \
  --output_dir "${OUTPUT_DIR}"
