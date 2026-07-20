#!/bin/bash
#SBATCH --job-name=verify_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=00:20:00
#SBATCH --output=./logs/verify_env_job-%j.out
#SBATCH --error=./logs/verify_env_job-%j.err

# Sottomesso in automatico da Setup_env.sh (o a mano: sbatch verify_env_job.sh).
# Fa cio' che Setup_env.sh non puo' fare sul nodo di login: importare torch.
# Sul login node di Mufasa, `import torch` (build cu130) puo' fallire con
#   ImportError: libtorch_cpu.so: failed to map segment from shared object
# per limiti di memoria della sessione interattiva. Qui, dentro un job, quei
# limiti non si applicano.
#
# In piu' (se omnivggt e' installato), costruisce OmniVGGT() una volta sola:
# Aggregator.__init__ scarica incondizionatamente il backbone DINOv2 via
# torch.hub SENZA try/except — farlo qui (nodo con presumibilmente accesso a
# internet quanto il login node, entrambi sullo stesso cluster) pre-popola la
# cache in $TORCH_HOME cosi' le run vere non ripetono il download.

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "  Verify env (torch / spectra / omnivggt)"
echo "  Job ID  : ${SLURM_JOB_ID:-<manuale>}"
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
    echo "ERRORE: uv non trovato. Esegui prima:  bash Setup_env.sh"
    exit 1
fi
if [ ! -d ".venv" ]; then
    echo "ERRORE: .venv non trovato. Esegui prima:  bash Setup_env.sh"
    exit 1
fi

echo "--- Smoke test base ---------------------------------------------------"
uv run python -c "
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
import spectra
print(f'  spectra   : {spectra.__version__}')
"

echo "--- omnivggt ------------------------------------------------------------"
uv run python -c "
try:
    from omnivggt.models.omnivggt import OmniVGGT
except ImportError:
    print('  omnivggt  : non installato (vggt.impl=omni non disponibile)')
else:
    print('  omnivggt  : importabile, costruisco OmniVGGT() (pre-download DINOv2)...')
    OmniVGGT()
    print('  omnivggt  : OK, DINOv2 backbone in cache (vggt.impl=omni disponibile)')
"

echo "=============================================="
echo "  Verifica completata: $(date)"
echo "  Ora puoi lanciare: sbatch run_spectra.sh"
echo "=============================================="
