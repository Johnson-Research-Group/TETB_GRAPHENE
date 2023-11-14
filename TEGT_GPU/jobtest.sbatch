#!/bin/bash
#SBATCH --partition qmchamm
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --time 72:00:00
#SBATCH --output test_intralayer.log
#SBATCH --job-name test_intralayer
#SBATCH --mail-user dpalmer3@illinois.edu
#SBATCH --mail-type ALL
module load anaconda

source activate my.anaconda
python fit_potentials.py -m popov -t intralayer -k 225 -g False -f False -s True 
