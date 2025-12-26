#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate the Python 3.12 conda environment.
if [[ -f "/workspace/aimo/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/workspace/aimo/miniconda/etc/profile.d/conda.sh"
  conda activate tensorrt-3.12
else
  echo "ERROR: conda.sh not found; cannot activate tensorrt-3.12 env." >&2
  exit 1
fi

# Respect preinstalled packages and avoid pip overrides by default.
export TRTLLM_SKIP_REQUIREMENTS="${TRTLLM_SKIP_REQUIREMENTS:-1}"

# Point build to CUDA 12.8.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Normalize CLI flags: build_wheel.py expects --cuda_architectures.
args=()
ucx_override=""
trt_root_arg=""
nccl_root_arg=""
cuda_root_override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-architectures=*)
      args+=("--cuda_architectures=${1#*=}")
      ;;
    --cuda-architectures)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --cuda-architectures expects a value." >&2
        exit 1
      fi
      args+=("--cuda_architectures=${1}")
      ;;
    --trt_root=*)
      trt_root_arg="${1#*=}"
      args+=("$1")
      ;;
    --trt_root)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --trt_root expects a value." >&2
        exit 1
      fi
      trt_root_arg="$1"
      args+=("--trt_root" "$1")
      ;;
    --trt-root=*)
      trt_root_arg="${1#*=}"
      args+=("--trt_root=${1#*=}")
      ;;
    --trt-root)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --trt-root expects a value." >&2
        exit 1
      fi
      trt_root_arg="$1"
      args+=("--trt_root" "$1")
      ;;
    --nccl_root=*)
      nccl_root_arg="${1#*=}"
      args+=("$1")
      ;;
    --nccl_root)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --nccl_root expects a value." >&2
        exit 1
      fi
      nccl_root_arg="$1"
      args+=("--nccl_root" "$1")
      ;;
    --nccl-root=*)
      nccl_root_arg="${1#*=}"
      args+=("--nccl_root=${1#*=}")
      ;;
    --nccl-root)
      shift
      if [[ -z "${1:-}" ]]; then
        echo "ERROR: --nccl-root expects a value." >&2
        exit 1
      fi
      nccl_root_arg="$1"
      args+=("--nccl_root" "$1")
      ;;
    *)
      if [[ "$1" == *"ENABLE_UCX"* ]]; then
        ucx_override="1"
      fi
      if [[ "$1" == *"CUDAToolkit_ROOT="* ]]; then
        cuda_root_override="1"
      fi
      args+=("$1")
      ;;
  esac
  shift
done

if [[ -z "${ucx_override}" ]]; then
  args+=("--extra-cmake-vars" "ENABLE_UCX=OFF")
fi
if [[ -z "${cuda_root_override}" && -d "${CUDA_HOME}" ]]; then
  args+=("--extra-cmake-vars" "CUDAToolkit_ROOT=${CUDA_HOME}")
fi

# Default to Kaggle TensorRT dataset if no TRT root is provided.
if [[ -z "${trt_root_arg}" && -z "${TRT_ROOT:-}" ]]; then
  dataset_id="${TRT_KAGGLE_DATASET_ID:-mnhhang/tensorrt-10-14-1-48}"
  dataset_version="${TRT_KAGGLE_DATASET_VERSION:-1}"
  dataset_dir="${TRT_KAGGLE_TRT_DIRNAME:-TensorRT-10.14.1.48}"
  cache_root="${KAGGLEHUB_CACHE:-$HOME/.cache/kagglehub}"
  cached_root="${cache_root}/datasets/${dataset_id}/versions/${dataset_version}/${dataset_dir}"

  if [[ -d "${cached_root}/lib" && -d "${cached_root}/python" ]]; then
    TRT_ROOT="${cached_root}"
  else
    echo "[build_trtllm_py312_conda] Downloading TensorRT dataset from Kaggle" >&2
    tmp_root_file="$(mktemp)"
    KAGGLEHUB_DISABLE_PROGRESS=1 python - <<'PY' "${tmp_root_file}"
import os
import pathlib
import sys

try:
    import kagglehub
except Exception as exc:
    raise SystemExit(
        "kagglehub is required to download TensorRT. Install it with: pip install kagglehub"
    ) from exc

dataset = os.environ.get("TRT_KAGGLE_DATASET_ID", "mnhhang/tensorrt-10-14-1-48")
path = kagglehub.dataset_download(dataset)
pathlib.Path(sys.argv[1]).write_text(path)
PY
    TRT_ROOT="$(cat "${tmp_root_file}")"
    rm -f "${tmp_root_file}"
    if [[ -d "${TRT_ROOT}/${dataset_dir}/lib" ]]; then
      TRT_ROOT="${TRT_ROOT}/${dataset_dir}"
    fi
  fi
  export TRT_ROOT
fi

if [[ -n "${TRT_ROOT:-}" && -z "${trt_root_arg}" ]]; then
  TRT_SDK="${TRT_SDK:-${ROOT_DIR}/build/trt_sdk}"
  mkdir -p "${TRT_SDK}/lib"
  ln -sfn "${TRT_ROOT}/include" "${TRT_SDK}/include"
  for base in libnvinfer libnvinfer_plugin libnvonnxparser libnvinfer_lean libnvinfer_dispatch libnvinfer_vc_plugin; do
    versioned_so=""
    for candidate in "${TRT_ROOT}/lib/${base}.so."*; do
      if [[ -f "${candidate}" ]]; then
        versioned_so="${candidate}"
        break
      fi
    done
    if [[ -n "${versioned_so}" ]]; then
      ln -sf "${versioned_so}" "${TRT_SDK}/lib/${base}.so.10"
      ln -sf "${versioned_so}" "${TRT_SDK}/lib/${base}.so"
    fi
  done
  export TRT_ROOT="${TRT_SDK}"
  args+=("--trt_root" "${TRT_ROOT}")
  export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${TRT_ROOT}/lib:${LIBRARY_PATH:-}"
  export CMAKE_PREFIX_PATH="${TRT_ROOT}:${CMAKE_PREFIX_PATH:-}"
fi

# Use NCCL from pip wheels if available to satisfy newer symbols.
if [[ -z "${nccl_root_arg}" ]]; then
  NCCL_ROOT="$(python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / "nccl"
    if (candidate / "lib").is_dir():
        print(candidate)
        break
PY
)"
  if [[ -n "${NCCL_ROOT}" ]]; then
    export NCCL_ROOT
    args+=("--nccl_root" "${NCCL_ROOT}")
    export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
    export CMAKE_PREFIX_PATH="${NCCL_ROOT}:${CMAKE_PREFIX_PATH:-}"
  fi
fi

# Prefer UCX from conda when available to satisfy newer UCXX requirements.
if [[ -z "${UCX_ROOT:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  if [[ -f "${CONDA_PREFIX}/lib/cmake/ucx/ucx-config.cmake" ]]; then
    export UCX_ROOT="${CONDA_PREFIX}"
    export LD_LIBRARY_PATH="${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${UCX_ROOT}/lib:${LIBRARY_PATH:-}"
    export CMAKE_PREFIX_PATH="${UCX_ROOT}:${CMAKE_PREFIX_PATH:-}"
  fi
fi

# Use the active environment without creating a nested venv.
exec python "${ROOT_DIR}/scripts/build_wheel.py" --no-venv "${args[@]}"
