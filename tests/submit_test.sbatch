#!/bin/bash
#SBATCH --account m4205
#SBATCH --qos debug
#SBATCH --job-name test
#SBATCH --constraint cpu
#SBATCH --nodes=1
#SBATH --ntasks-per-node=1
#SBATCH --time=00:20:00
##SBATCH --ntasks-per-node=4
##SBATCH --gpus-per-node=4
module load python/3.9-anaconda-2021.11
module load julia
module load cudatoolkit/11.7
source activate $PSCRATCH/mypythonev

#srun python mpi_ase_calc.py
srun python test_relax.py
