#!/bin/bash
#SBATCH --job-name=build_spectra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64g
#SBATCH --partition=jobs
#SBATCH --qos=nogpu
#SBATCH --time=06:00:00
#SBATCH --output=./logs/build_sbatch-%j.out
#SBATCH --error=./logs/build_sbatch-%j.err
set -euo pipefail
mkdir -p ./logs

echo "=============================================="
echo "  Spectra container build"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=============================================="

# --- Cartelle temporanee NELLA HOME (NON in /tmp e NON in /home/shared) ------
# Singularity usa TMPDIR per estrarre i layer e CACHEDIR per la cache immagini.
# Devono stare nella home dell'utente:
#   - /tmp        -> vietato (rischia di saturare la root di sistema)
#   - /home/shared -> per dataset condivisi, quasi pieno e senza permessi di scrittura per la build
#   - $HOME       -> corretto
BUILD_DIR="${HOME}/singularity_build_${SLURM_JOB_ID}"
export SINGULARITY_TMPDIR="${BUILD_DIR}/tmp"
export SINGULARITY_CACHEDIR="${BUILD_DIR}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
echo "  TMPDIR  : ${SINGULARITY_TMPDIR}"
echo "  CACHEDIR: ${SINGULARITY_CACHEDIR}"

# Pulizia garantita anche in caso di errore o interruzione del job ------------
cleanup() {
    echo "  Pulizia cartella temporanea: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

# --- Check spazio disponibile nella home -------------------------------------
# Una build con CUDA + dipendenze puo' generare decine di GB di temporanei.
# Si ferma subito se ci sono meno di MIN_GB liberi, invece di riempire la home a meta'.
MIN_GB=40
AVAIL_GB=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
echo "  Spazio libero in ${HOME}: ${AVAIL_GB} GB (minimo richiesto: ${MIN_GB} GB)"
if [ "${AVAIL_GB}" -lt "${MIN_GB}" ]; then
    echo "ERRORE: spazio insufficiente nella home per la build."
    echo "Libera spazio oppure costruisci l'immagine sul tuo PC e trasferiscila con scp/rsync."
    exit 1
fi

# --- Load Singularity --------------------------------------------------------
module load amd/singularity

# --- Build -------------------------------------------------------------------
# --fakeroot : build come utente non-root (richiesto sulla maggior parte degli HPC)
# Nessun --nv qui: CUDA e' baked nell'image via il docker base,
# la GPU dell'host non serve in fase di build.
singularity build --nv --fakeroot spectra.sif spectra.def

echo "=============================================="
echo "  Build finished : $(date)"
echo "  Image size     : $(du -sh spectra.sif | cut -f1)"
echo "=============================================="