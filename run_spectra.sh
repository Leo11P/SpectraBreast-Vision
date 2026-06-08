#!/bin/bash
#SBATCH --job-name=spectra_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --partition=jobs
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=24:00:00
#SBATCH --output=./logs/run_spectra-%j.out
#SBATCH --error=./logs/run_spectra-%j.err

set -euo pipefail
mkdir -p logs

module load amd/singularity

# Monta DATA e RESULTS dal tuo home dentro il container
singularity exec --nv \
    --bind /home/lpassoni/SpectraBreast-Vision/DATA:/data/DATA \
    --bind /home/lpassoni/SpectraBreast-Vision/RESULTS:/data/RESULTS \
    --bind /home/lpassoni/SpectraBreast-Vision/configs:/data/configs \
    --bind /home/lpassoni/mast3r_checkpoints:/opt/mast3r/checkpoints \
    spectra.sif 
    spectra full \
        --config /data/configs/default.yaml \
        --sample SAMPLE1 \
        -s rerun.enabled=false