#!/bin/bash
#SBATCH --job-name=build_spectra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --partition=jobs
#SBATCH --qos=build
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

# --- Load Singularity --------------------------------------------------------
module load amd/singularity

# --- Build -------------------------------------------------------------------
# --fakeroot  : build as non-root user (required on most HPC clusters)
# No --nv here: CUDA is baked into the image via the docker base,
# the host GPU is not needed at build time.
singularity build --fakeroot spectra.sif spectra.def

echo "=============================================="
echo "  Build finished : $(date)"
echo "  Image size     : $(du -sh spectra.sif | cut -f1)"
echo "=============================================="
