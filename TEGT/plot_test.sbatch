#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 12:00:00
#SBATCH --account m4205
#SBATCH --qos regular
#SBATCH --job-name interlayer_plot
#SBATCH --constraint cpu
#SBATCH --output interlayer_plot.log
module load python/3.9-anaconda-2021.11
module load julia
source activate $PSCRATCH/mypythonev
python fit_potentials.py -m popov -t interlayer -k 1 -f False -s True
python fit_potentials.py -m popov -t interlayer -k 225 -f False -s True
