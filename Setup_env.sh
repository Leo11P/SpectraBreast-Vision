#!/bin/bash
# setup_env.sh — da eseguire UNA VOLTA sul nodo di login di Mufasa (NON via sbatch)
# Crea il virtual environment uv usando vggt-omega gia' clonato DENTRO la repo.
#
# Uso:
#   cd <repo>            # ovunque tu abbia clonato SpectraBreast-Vision
#   bash Setup_env.sh

set -euo pipefail

echo "=============================================="
echo "  Setup ambiente Spectra (uv, no Singularity)"
echo "  User    : ${USER}"
echo "  Start   : $(date)"
echo "=============================================="

# Radice repo = cartella che contiene questo script (funziona ovunque sia clonata).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
echo "  Repo    : ${REPO_DIR}"

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

# --- 2. Verifica che vggt-omega sia gia' clonato DENTRO la repo --------------
# (clonalo con:  git clone https://github.com/facebookresearch/vggt-omega.git)
# questo script NON lo riclona, lo usa dove si trova.
VGGT_DIR="${REPO_DIR}/vggt-omega"

if [ ! -d "${VGGT_DIR}" ]; then
    echo "ERRORE: ${VGGT_DIR} non trovato. Clona vggt-omega dentro la repo prima di procedere:"
    echo "        git clone https://github.com/facebookresearch/vggt-omega.git"
    exit 1
fi
echo "[2/4] vggt-omega trovato dentro la repo."

# --- 3. uv sync (crea .venv e installa le dipendenze del pyproject) ----------
echo "[3/4] uv sync (scarica PyTorch cu130, puo' volerci qualche minuto)..."
uv sync

# Installa vggt-omega (editable) e le sue dipendenze nello stesso .venv.
# NB: si usa `uv pip` (il pip integrato di uv), NON `uv run pip`: la venv creata
# da `uv sync` non contiene un eseguibile `pip`.
if [ -f "${VGGT_DIR}/requirements.txt" ]; then
    uv pip install --no-cache-dir -r "${VGGT_DIR}/requirements.txt"
fi
uv pip install --no-cache-dir -e "${VGGT_DIR}"

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