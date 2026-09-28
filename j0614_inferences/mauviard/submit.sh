#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="./outdir/log.out"
#SBATCH --job-name="mauviard"

now=$(date)
echo "$now"

# Run jester
source /home/twouters2/projects/49_strange_quark_stars/.venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
run_jester_inference config.yaml