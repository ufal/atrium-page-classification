#!/bin/bash
#SBATCH -J clip  # name of job
#SBATCH -p gpu-ms,gpu-troja        # name of partition or queue (if not specified default partition is used)
#SBATCH --cpus-per-task=16                 # number of cores/threads per task (default 1)
#SBATCH --gres=gpu:1
#SBATCH --mem=50G                         # request 16 gigabytes memory (per node, default depends on node)
#SBATCH --time 5-00:00:00 # time (D-HH:MM:SS)
#SBATCH -o clip.%A_%a.%N.out  # name of output file for this submission script
#SBATCH -e clip.%A_%a.%N.err
hostname
date

export HF_HOME=/lnet/work/projects/atrium/cache/
export TORCH_HOME=/lnet/work/projects/atrium/cache/

source /net/work/projects/atrium/venv-atrium/bin/activate

cd /net/work/projects/atrium/clip

ARG=( "$@" )

python3 ./run.py ${ARG[*]}

date
