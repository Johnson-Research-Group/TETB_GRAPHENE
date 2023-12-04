#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 12:00:00
#SBATCH --account m4205
#SBATCH --qos regular
#SBATCH --job-name nkp1_intralayer
#SBATCH --constraint cpu
#SBATCH --output nkp1_intralayer.log
module load python/3.9-anaconda-2021.11

source activate $PSCRATCH/mypythonev
python fit_potentials.py -m popov -t intralayer -k 1
