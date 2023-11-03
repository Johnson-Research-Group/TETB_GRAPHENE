#!/bin/bash
#SBATCH --partition qmchamm
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --time 72:00:00
#SBATCH --output mesh1_intralayer.log
#SBATCH --job-name mesh1_intralayer
#SBATCH --mail-user dpalmer3@illinois.edu
#SBATCH --mail-type ALL
module load anaconda

source activate my.anaconda
python fit_potentials.py -m popov -t intralayer -k 1 -g False -oz tb_weight
