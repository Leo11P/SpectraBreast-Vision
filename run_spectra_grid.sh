#!/bin/bash
#SBATCH --job-name=spectra_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=5:00:00
#SBATCH --output=./logs/grid_spectra-%j.out
#SBATCH --error=./logs/grid_spectra-%j.err

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "  Spectra GRID SEARCH (uv, no Singularity)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=============================================="

# --- Repo root (sbatch copies the script to spool; use SLURM_SUBMIT_DIR) ------
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"
echo "  Repo    : ${REPO_DIR}"

# --- Keep ALL caches in home (never /tmp) ------------------------------------
export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

export PATH="${HOME}/.local/bin:${PATH}"

# --- Pre-flight --------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "ERRORE: uv non trovato. Esegui prima:  bash Setup_env.sh"
    exit 1
fi
if [ ! -d ".venv" ]; then
    echo "ERRORE: .venv non trovato. Esegui prima:  bash Setup_env.sh"
    exit 1
fi
echo "  Python  : $(uv run python --version)"
echo "  Torch   : $(uv run python -c 'import torch; print(torch.__version__)')"
echo "  CUDA ok : $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================================="

# --- Launch the grid-search driver -------------------------------------------
# The driver calls `spectra full` once per combo with --force-vision, samples
# GPU memory, times each run, parses metrics, and writes the report incrementally
# to RESULTS/grid_search/ (so a mid-grid crash still leaves a partial report).

uv run --no-sync python grid_search_vision.py \
    --config configs/default.yaml \
    --sample SAMPLE2 \
    --out-dir RESULTS/grid_search
#uv run --no-sync python grid_search_vision.py --dry-run

echo "=============================================="
echo "  Grid finished : $(date)"
echo "=============================================="
