#!/bin/bash
#SBATCH --job-name=spectra_reject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/reject_sweep-%j.out
#SBATCH --error=./logs/reject_sweep-%j.err

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "  Spectra REJECT-THRESHOLD SWEEP"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=============================================="

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"
echo "  Repo    : ${REPO_DIR}"

export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv &> /dev/null; then
    echo "ERRORE: uv non trovato. Esegui prima:  bash Setup_env.sh"; exit 1
fi
if [ ! -d ".venv" ]; then
    echo "ERRORE: .venv non trovato. Esegui prima:  bash Setup_env.sh"; exit 1
fi
echo "  Python  : $(uv run python --version)"
echo "=============================================="

uv run --no-sync python grid_search_reject.py \
    --config configs/default.yaml \
    --sample SAMPLE2 \
    --out-dir RESULTS/reject_sweep

echo "=============================================="
echo "  Sweep finished : $(date)"
echo "=============================================="
