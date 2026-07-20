#!/bin/bash
# setup_env.sh — da eseguire UNA VOLTA sul nodo di login di Mufasa (NON via sbatch)
# Crea il virtual environment uv usando vggt-omega gia' clonato DENTRO la repo.
# Se e' presente anche omnivggt (per vggt.impl=omni), lo registra nella venv.
#
# Questo script fa SOLO operazioni leggere (download pacchetti, .pth file) e
# non esegue mai `import torch`: sul nodo di login di Mufasa, il solo import
# di torch (build cu130) puo' fallire con
#   ImportError: libtorch_cpu.so: failed to map segment from shared object
# per limiti di memoria della sessione interattiva che non e' possibile
# alzare. La verifica vera (torch/spectra/omnivggt importabili + pre-download
# dei pesi DINOv2 richiesti da OmniVGGT) gira invece come job SLURM, dove tali
# limiti non si applicano — vedi verify_env_job.sh, lanciato in automatico
# alla fine di questo script.
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
    echo "[1/5] Installazione uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "  uv installato: $(uv --version)"
else
    echo "[1/5] uv gia' presente: $(uv --version)"
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
echo "[2/5] vggt-omega trovato dentro la repo."

# --- 3. omnivggt (opzionale: solo se vuoi usare vggt.impl=omni) --------------
# (clonalo con:  git clone https://github.com/Livioni/OmniVGGT-official.git omnivggt)
# A differenza di vggt-omega non e' obbligatorio: se manca, il setup continua e
# semplicemente 'omni' non sara' disponibile come vggt.impl.
OMNIVGGT_DIR="${REPO_DIR}/omnivggt"
HAVE_OMNIVGGT=0
if [ -d "${OMNIVGGT_DIR}" ]; then
    HAVE_OMNIVGGT=1
    echo "[3/5] omnivggt trovato dentro la repo (vggt.impl=omni sara' disponibile)."
else
    echo "[3/5] omnivggt non trovato: salto (vggt.impl=omni NON sara' disponibile)."
    echo "      Per abilitarlo: git clone https://github.com/Livioni/OmniVGGT-official.git omnivggt"
fi

# --- 4. uv sync (crea .venv e installa le dipendenze del pyproject) ----------
echo "[4/5] uv sync (scarica PyTorch cu130, puo' volerci qualche minuto)..."
uv sync

# Installa vggt-omega (editable) e le sue dipendenze nello stesso .venv.
# NB: si usa `uv pip` (il pip integrato di uv), NON `uv run pip`: la venv creata
# da `uv sync` non contiene un eseguibile `pip`.
if [ -f "${VGGT_DIR}/requirements.txt" ]; then
    uv pip install --no-cache-dir -r "${VGGT_DIR}/requirements.txt"
fi
uv pip install --no-cache-dir -e "${VGGT_DIR}"

if [ "${HAVE_OMNIVGGT}" -eq 1 ]; then
    # OmniVGGT-official NON e' un pacchetto pip installabile: alla radice non
    # ha ne' pyproject.toml ne' setup.py, solo cartelle sciolte (omnivggt/,
    # inference.py, ...). `uv pip install -e` fallisce sempre qui ("does not
    # appear to be a Python project") — non e' un problema di versione, manca
    # proprio il packaging. La soluzione standard per repo fatte cosi' e' un
    # file .pth nella venv: aggiunge OMNIVGGT_DIR a sys.path (il modulo
    # `omnivggt/` sotto e' importabile come namespace package Python 3, non
    # richiede __init__.py).
    SITE_PACKAGES="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "${OMNIVGGT_DIR}" > "${SITE_PACKAGES}/omnivggt_local.pth"
    echo "  omnivggt registrato via .pth in ${SITE_PACKAGES}"

    # `omnivggt.utils.geometry` (closed_form_inverse_se3, usata davvero dal
    # nostro backend) importa `omnivggt.utils.misc` -> `omnivggt.utils.vo_eval`,
    # che a livello di modulo fa `import evo` INCONDIZIONATAMENTE — solo per
    # metriche di traiettoria che non chiamiamo mai, ma l'import fallisce
    # comunque se il pacchetto manca. E' l'unica dipendenza reale non gia'
    # coperta dal resto del progetto (verificato leggendo tutta la sotto-catena
    # geometry/misc/vo_eval/device: nient'altro di nuovo oltre a questa).
    # Il resto del requirements.txt di OmniVGGT (viser, onnxruntime,
    # transformers, mmengine, gradio==5.17.1, pydantic==2.10.6...) resta
    # inutile per noi e non lo installiamo (rischierebbe di retrocedere
    # gradio/pydantic che il resto della pipeline usa in versioni piu' nuove).
    uv pip install --no-cache-dir evo
fi

# --- 5. Verifica (torch/spectra/omnivggt + pre-download DINOv2) via sbatch ---
# Deliberatamente NON un `uv run python -c "import torch"` qui: vedi il
# commento in testa al file. Il job stampa il suo output in logs/.
echo "[5/5] Sottometto il job di verifica (verify_env_job.sh)..."
mkdir -p logs
JOB_ID=$(sbatch --parsable verify_env_job.sh)
echo "  Job sottomesso: ${JOB_ID}"
echo "  Segui l'output con:  tail -f logs/verify_env_job-${JOB_ID}.out"

echo "=============================================="
echo "  Setup (parte login-node) completato: $(date)"
echo "  Controlla l'esito del job ${JOB_ID}, poi lancia: sbatch run_spectra.sh"
echo "=============================================="