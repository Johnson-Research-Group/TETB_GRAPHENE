#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --account=bcmp-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --job-name=generate_ensembles
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=208g
#SBATCH --output=generate_ensembles.log
#SBATCH --ntasks-per-node=1
module load anaconda3_gpu/23.9.0

source activate myenv
export OMP_NUM_THREADS=1
python generate_ensemble_models.py
