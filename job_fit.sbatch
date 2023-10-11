#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --account=m4205
#SBATCH --qos=regular
#SBATCH --job-name=fit_kcinsp
#SBATCH --constraint=cpu
module load gcc
module load python/3.9-anaconda-2021.11

source activate $PSCRATCH/mypythonev
srun python  fit_potentials.py
