#!/bin/bash
#SBATCH --job-name=spectra_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=6:00:00
#SBATCH --output=./logs/run_spectra-%j.out
#SBATCH --error=./logs/run_spectra-%j.err

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "  Spectra run (uv, no Singularity)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=============================================="

# --- Vai nella root della repo (= cartella di questo script) ----------------
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"
echo "  Repo    : ${REPO_DIR}"

# --- Tieni TUTTE le cache nella home (mai in /tmp) ---------------------------
# uv, pip, huggingface e torch scrivono cache temporanee: senza queste righe
# alcune finirebbero in /tmp. Le forziamo nella home per non saturare /.
export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

# --- Assicura uv nel PATH (lo installa setup_env.sh in ~/.local/bin) ---------
export PATH="${HOME}/.local/bin:${PATH}"

# --- Verifiche pre-volo ------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "ERRORE: uv non trovato. Esegui prima:  bash setup_env.sh"
    exit 1
fi
if [ ! -d ".venv" ]; then
    echo "ERRORE: .venv non trovato. Esegui prima:  bash setup_env.sh"
    exit 1
fi

echo "  Python  : $(uv run python --version)"
echo "  Torch   : $(uv run python -c 'import torch; print(torch.__version__)')"
echo "  CUDA ok : $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "  TMPDIR  : ${TMPDIR}"
echo "=============================================="

# --- Lancia la pipeline ------------------------------------------------------
# Sample e tutte le path sono definiti dentro configs/default.yaml,
# quindi NON serve passare --sample qui (eviti di specificarlo in due posti).

#uv run spectra full \
#    --config configs/default.yaml \
#    -s vision.rerun.enabled=false

#uv run spectra registration \
#    --config configs/default.yaml \
#    --sample SAMPLE2
 
#uv run spectra full \
#     --config configs/default.yaml \
#     --force-vision \
#     --sample SAMPLE2

uv run spectra vision --config configs/default.yaml

echo "=============================================="
echo "  Run finished : $(date)"
echo "=============================================="