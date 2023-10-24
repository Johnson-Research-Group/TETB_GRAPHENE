#!/bin/bash
#SBATCH --partition qmchamm
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --time 72:00:00
#SBATCH --output logout.test_interlayer
#SBATCH --job-name test_interlayer
#SBATCH --mail-user dpalmer3@illinois.edu
#SBATCH --mail-type ALL
module load anaconda

source activate my.anaconda
#python fit_potentials.py -m popov -t interlayer -k 225 -s True
python fit_potentials.py -s True -t interlayer -f False
