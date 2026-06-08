#!/bin/bash
# setup_env.sh — da eseguire UNA VOLTA sul nodo di login di Mufasa (NON via sbatch)
# Crea il virtual environment uv usando mast3r e asmk gia' clonati DENTRO la repo.
#
# Uso:
#   cd ~/SpectraBreast-Vision
#   bash setup_env.sh

set -euo pipefail

echo "=============================================="
echo "  Setup ambiente Spectra (uv, no Singularity)"
echo "  User    : ${USER}"
echo "  Start   : $(date)"
echo "=============================================="

REPO_DIR="${HOME}/SpectraBreast-Vision"
cd "${REPO_DIR}"

# --- Cache nella home (mai in /tmp): qui uv scarica i GB di PyTorch ----------
export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

# --- Check spazio disponibile nella home -------------------------------------
MIN_GB=20
AVAIL_GB=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
echo "  Spazio libero in ${HOME}: ${AVAIL_GB} GB (minimo richiesto: ${MIN_GB} GB)"
if [ "${AVAIL_GB}" -lt "${MIN_GB}" ]; then
    echo "ERRORE: spazio insufficiente nella home. Libera spazio prima di procedere."
    exit 1
fi

# --- 1. Installa uv (se non c'e' gia') ---------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv &> /dev/null; then
    echo "[1/4] Installazione uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "  uv installato: $(uv --version)"
else
    echo "[1/4] uv gia' presente: $(uv --version)"
fi

# --- 2. Verifica che mast3r e asmk siano gia' clonati DENTRO la repo ---------
# (li hai gia' clonati: questo script NON li riclona, li usa dove sono)
MAST3R_DIR="${REPO_DIR}/mast3r"
ASMK_DIR="${REPO_DIR}/asmk"

if [ ! -d "${MAST3R_DIR}" ]; then
    echo "ERRORE: ${MAST3R_DIR} non trovato. Clona mast3r dentro la repo prima di procedere."
    exit 1
fi
if [ ! -d "${ASMK_DIR}" ]; then
    echo "ERRORE: ${ASMK_DIR} non trovato. Clona asmk dentro la repo prima di procedere."
    exit 1
fi
echo "[2/4] mast3r e asmk trovati dentro la repo."

# --- 3. uv sync (crea .venv e installa le dipendenze del pyproject) ----------
echo "[3/4] uv sync (scarica PyTorch cu130, puo' volerci qualche minuto)..."
uv sync

# Installa le dipendenze di mast3r/dust3r e asmk dentro lo stesso .venv,
# leggendole dai cloni gia' presenti nella repo.
uv run pip install --no-cache-dir -r "${MAST3R_DIR}/requirements.txt"
if [ -f "${MAST3R_DIR}/dust3r/requirements.txt" ]; then
    uv run pip install --no-cache-dir -r "${MAST3R_DIR}/dust3r/requirements.txt"
fi
uv run pip install --no-cache-dir -e "${ASMK_DIR}"

# --- 4. Smoke test -----------------------------------------------------------
echo "[4/4] Smoke test..."
uv run python -c "
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
import spectra
print(f'  spectra   : {spectra.__version__}')
print('  OK')
"

echo "=============================================="
echo "  Setup completato: $(date)"
echo "  Ora puoi lanciare: sbatch run_spectra.sh"
echo "=============================================="