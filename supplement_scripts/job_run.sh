#!/bin/bash
#SBATCH -J lang  # name of job
#SBATCH -p gpu-ms,gpu-troja        # name of partition or queue (if not specified default partition is used)
#SBATCH --cpus-per-task=32                 # number of cores/threads per task (default 1)
#SBATCH --gres=gpu:1                       # number of GPUs to request (default 0)
#SBATCH --mem=50G                         # request 16 gigabytes memory (per node, default depends on node)
#SBATCH --time 3-00:00:00 # time (D-HH:MM:SS)
#SBATCH -o lang.%A_%a.%N.out  # name of output file for this submission script
#SBATCH -e lang.%A_%a.%N.err

hostname
date

# --- REFINED: Portability using Slurm environment variables ---
# Dynamically assigns the project root based on where the job was submitted from.
# Uses a fallback to the original absolute path if SLURM_SUBMIT_DIR is not set.
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-/lnet/work/projects/atrium}"

export HF_HOME="${PROJECT_DIR}/cache/"
export TORCH_HOME="${PROJECT_DIR}/cache/"

source "${PROJECT_DIR}/alto_util/venv-lang/bin/activate"

cd "${PROJECT_DIR}/alto_util" || exit

ARG=( "$@" )

python3 langID.py "${ARG[@]}"

date